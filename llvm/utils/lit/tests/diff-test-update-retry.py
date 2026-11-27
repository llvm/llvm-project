# RUN: rm -rf %t && mkdir -p %t

# RUN: cp %S/Inputs/diff-test-update-retry/1.in %t/1.in
# RUN: cp %S/Inputs/diff-test-update-retry/lit.cfg %t/lit.cfg
#
# RUN: cp %S/Inputs/diff-test-update-retry/single-split-file.in %t/single-split-file.test
# RUN: cp %S/Inputs/diff-test-update-retry/multiple-split-file-enough-retries.in %t/multiple-split-file-enough-retries.test
# RUN: cp %S/Inputs/diff-test-update-retry/multiple-split-file-not-enough-retries.in %t/multiple-split-file-not-enough-retries.test
# RUN: cp %S/Inputs/diff-test-update-retry/multiple-split-file-unrelated-failure.in %t/multiple-split-file-unrelated-failure.test

# RUN: not %{lit} --update-tests -v %t > %t/out.txt

# RUN: diff --strip-trailing-cr %S/Inputs/diff-test-update-retry/single-split-file.out %t/single-split-file.test
# RUN: diff --strip-trailing-cr %S/Inputs/diff-test-update-retry/multiple-split-file-enough-retries.out %t/multiple-split-file-enough-retries.test
# RUN: diff --strip-trailing-cr %S/Inputs/diff-test-update-retry/multiple-split-file-not-enough-retries.out %t/multiple-split-file-not-enough-retries.test
# RUN: diff --strip-trailing-cr %S/Inputs/diff-test-update-retry/multiple-split-file-unrelated-failure.out %t/multiple-split-file-unrelated-failure.test

# RUN: FileCheck %s --match-full-lines < %t/out.txt

# CHECK-LABEL: FIXED: diff-test-update-retry :: multiple-split-file-enough-retries.test (1 of 4, 5 of 6 attempts)
# CHECK-NEXT: [Attempt 1]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test2.expected in {{.*}}multiple-split-file-enough-retries.test
# CHECK-NEXT: [Attempt 2]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test3.expected in {{.*}}multiple-split-file-enough-retries.test
# CHECK-NEXT: [Attempt 3]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test4.expected in {{.*}}multiple-split-file-enough-retries.test
# CHECK-NEXT: [Attempt 4]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test5.expected in {{.*}}multiple-split-file-enough-retries.test
# CHECK-NEXT: ********************

# CHECK-LABEL: FAIL: diff-test-update-retry :: multiple-split-file-not-enough-retries.test (2 of 4, 3 of 3 attempts)
# CHECK-NEXT: ******************** TEST 'diff-test-update-retry :: multiple-split-file-not-enough-retries.test' FAILED ********************
# CHECK:      **********
# CHECK-NEXT: [Attempt 1]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test2.expected in {{.*}}multiple-split-file-not-enough-retries.test
# CHECK-NEXT: [Attempt 2]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test3.expected in {{.*}}multiple-split-file-not-enough-retries.test
# CHECK-NEXT: [Attempt 3]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test4.expected in {{.*}}multiple-split-file-not-enough-retries.test
# CHECK-NEXT: ********************

# CHECK-LABEL: FAIL: diff-test-update-retry :: multiple-split-file-unrelated-failure.test (3 of 4, 5 of 5 attempts)
# CHECK-NEXT: ******************** TEST 'diff-test-update-retry :: multiple-split-file-unrelated-failure.test' FAILED ********************
# CHECK:      **********
# CHECK-NEXT: [Attempt 1]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test2.expected in {{.*}}multiple-split-file-unrelated-failure.test
# CHECK-NEXT: [Attempt 2]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test3.expected in {{.*}}multiple-split-file-unrelated-failure.test
# CHECK-NEXT: [Attempt 3]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test4.expected in {{.*}}multiple-split-file-unrelated-failure.test
# CHECK-NEXT: [Attempt 4]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test5.expected in {{.*}}multiple-split-file-unrelated-failure.test
# CHECK-NEXT: ********************

# CHECK-LABEL: FIXED: diff-test-update-retry :: single-split-file.test (4 of 4, 2 of 2 attempts)
# CHECK-NEXT: [Attempt 1]
# CHECK-NEXT: update-diff-test: copied {{.*}}out.txt to slice test.expected in {{.*}}single-split-file.test
# CHECK-NEXT: ********************

# CHECK-NEXT: ********************
# CHECK-NEXT: Failed Tests (2):
# CHECK-NEXT:   diff-test-update-retry :: multiple-split-file-not-enough-retries.test
# CHECK-NEXT:   diff-test-update-retry :: multiple-split-file-unrelated-failure.test
