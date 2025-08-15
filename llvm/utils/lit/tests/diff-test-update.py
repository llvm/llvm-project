# RUN: not %{lit} --update-tests -v %S/Inputs/diff-test-update | FileCheck %s

# CHECK: # update-diff-test: could not deduce source and target from {{.*}}/Inputs/diff-test-update/1.in and {{.*}}/Inputs/diff-test-update/2.in
# CHECK: # update-diff-test: could not deduce source and target from {{.*}}/diff-test-update/Output/diff-bail2.test.tmp/1.txt and {{.*}}/diff-test-update/Output/diff-bail2.test.tmp/2.txt
# CHECK: # update-diff-test: copied {{.*}}/Output/diff-expected.test.tmp/my-file.txt to {{.*}}/Output/diff-expected.test.tmp/my-file.expected
# CHECK: # update-diff-test: copied {{.*}}/Output/diff-tmp-dir.test.tmp/1.txt to {{.*}}/Inputs/diff-test-update/empty.txt
# CHECK: # update-diff-test: copied {{.*}}/Inputs/diff-test-update/Output/diff-tmp.test.tmp.txt to {{.*}}/Inputs/diff-test-update/diff-t-out.txt


# CHECK: Failed: 5 (100.00%)
