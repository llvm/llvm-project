# Check the simple progress bar.

# FIXME: this test depends on order of tests
# RUN: rm -f %{inputs}/progress-bar/.lit_test_times.txt

# RUN: not %{lit} -j 1 -s %{inputs}/progress-bar > %t.out
# RUN: FileCheck < %t.out %s
#
# CHECK: Testing:
# CHECK: FAIL: progress-bar :: test-1.txt (1 of 4)
# CHECK: Testing:  0.. 10.. 20
# CHECK: FAIL: progress-bar :: test-2.txt (2 of 4)
# CHECK: Testing:  0.. 10.. 20.. 30.. 40..
# CHECK: FAIL: progress-bar :: test-3.txt (3 of 4)
# CHECK: Testing:  0.. 10.. 20.. 30.. 40.. 50.. 60.. 70
# CHECK: FAIL: progress-bar :: test-4.txt (4 of 4)
# CHECK: Testing:  0.. 10.. 20.. 30.. 40.. 50.. 60.. 70.. 80.. 90..
