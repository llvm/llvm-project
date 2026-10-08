## Check that malformed .lit_test_times.txt aborts discovery with a
## clear diagnostic.

# RUN: cp %{inputs}/malformed-test-times/lit_test_times %{inputs}/malformed-test-times/.lit_test_times.txt
# RUN: not %{lit-no-order-opt} %{inputs}/malformed-test-times > %t.out 2> %t.err
# RUN: FileCheck --allow-empty --check-prefix=OUT < %t.out %s
# RUN: FileCheck --check-prefix=ERR < %t.err %s

# OUT-NOT: -- Testing:

# ERR: fatal: found malformed timing data in
# ERR-SAME: ; remove the file to regenerate it
