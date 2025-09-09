# RUN: not %{lit} --update-tests -v %S/Inputs/diff-test-update | FileCheck %s

# CHECK: # update-diff-test: could not deduce source and target from {{.*}}1.in and {{.*}}2.in
# CHECK: # update-diff-test: could not deduce source and target from {{.*}}1.txt and {{.*}}2.txt
# CHECK: # update-diff-test: copied {{.*}}my-file.txt to {{.*}}my-file.expected
# CHECK: # update-diff-test: copied {{.*}}1.txt to {{.*}}empty.txt
# CHECK: # update-diff-test: copied {{.*}}diff-tmp.test.tmp.txt to {{.*}}diff-t-out.txt


# CHECK: Failed: 5 (100.00%)
