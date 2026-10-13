from faq_agent import check_escalation

test_cases = [
    ("How can I book a caregiver?", False),
    ("How can I check my bill?", False),
    ("Can I change my caregiver?", False),
    ("I want to reschedule my appointment", False),
    ("I am unhappy with the service", True),
    ("Can you tell me which medicine to take?", True),
    ("The patient is unconscious", True),
    ("I want to speak to a human representative", True),
    ("I was charged twice for my payment", True),
    ("Do you provide services on Mars?", False),
]

passed = 0
failed = 0

print("=" * 50)
print("SAATHI SNEHA CARE - ESCALATION TESTS")
print("=" * 50)

for i, (question, expected) in enumerate(test_cases, start=1):
    result = check_escalation(question)

    if isinstance(result, tuple) and len(result) >= 1:
        actual = result[0] is not None
    else: 
        actual = bool(result)

    success = actual == expected

    if success:
        passed += 1
    else:
        failed += 1

    print(f"\nTest {i}: {'PASS' if success else 'FAIL'}")
    print(f"Question: {question}")
    print(f"Expected escalation: {expected}")
    print(f"Actual escalation: {actual}")

    if isinstance(result, tuple) and result[0] is not None:
        print(f"Category: {result[0]}")

    print("-" * 50)

print("\nFINAL TEST SUMMARY")
print(f"Total tests: {len(test_cases)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")