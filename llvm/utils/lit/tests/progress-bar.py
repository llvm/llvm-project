# Check the simple progress bar.

# RUN: not %{lit} -s %{inputs}/progress-bar > %t.out
# RUN: FileCheck < %t.out %s
#
# CHECK: -- Testing: 4 tests, 1 workers --
# CHECK-NEXT: Testing:
# CHECK-NEXT: FAIL: progress-bar :: test-1.txt (1 of 4)
# CHECK-NEXT: Testing:
# CHECK-NEXT: FAIL: progress-bar :: test-2.txt (2 of 4)
# CHECK-NEXT: Testing:
# CHECK-NEXT: FAIL: progress-bar :: test-3.txt (3 of 4)
# CHECK-NEXT: Testing:
# CHECK-NEXT: FAIL: progress-bar :: test-4.txt (4 of 4)
# CHECK-NEXT: Testing:  0.. 10.. 20.. 30.. 40.. 50.. 60.. 70.. 80.. 90..
# CHECK-NEXT: ********************
# CHECK-NEXT: Failed Tests (4):
# CHECK-NEXT:   progress-bar :: test-1.txt
# CHECK-NEXT:   progress-bar :: test-2.txt
# CHECK-NEXT:   progress-bar :: test-3.txt
# CHECK-NEXT:   progress-bar :: test-4.txt
