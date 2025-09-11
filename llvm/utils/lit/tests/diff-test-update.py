# RUN: cp %S/Inputs/diff-test-update/single-split-file.in %S/Inputs/diff-test-update/single-split-file.test
# RUN: cp %S/Inputs/diff-test-update/single-split-file-populated.in %S/Inputs/diff-test-update/single-split-file-populated.test
# RUN: cp %S/Inputs/diff-test-update/multiple-split-file.in %S/Inputs/diff-test-update/multiple-split-file.test
# RUN: cp %S/Inputs/diff-test-update/multiple-split-file-populated.in %S/Inputs/diff-test-update/multiple-split-file-populated.test
# RUN: cp %S/Inputs/diff-test-update/single-split-file-no-expected.in %S/Inputs/diff-test-update/single-split-file-no-expected.test
# RUN: cp %S/Inputs/diff-test-update/split-c-comments.in %S/Inputs/diff-test-update/split-c-comments.test
# RUN: cp %S/Inputs/diff-test-update/split-whitespace.in "%S/Inputs/diff-test-update/split whitespace.test"

# RUN: not %{lit} --update-tests -v %S/Inputs/diff-test-update | FileCheck %s

# RUN: diff %S/Inputs/diff-test-update/single-split-file.out %S/Inputs/diff-test-update/single-split-file.test
# RUN: diff %S/Inputs/diff-test-update/single-split-file.out %S/Inputs/diff-test-update/single-split-file-populated.test
# RUN: diff %S/Inputs/diff-test-update/multiple-split-file.out %S/Inputs/diff-test-update/multiple-split-file.test
# RUN: diff %S/Inputs/diff-test-update/multiple-split-file.out %S/Inputs/diff-test-update/multiple-split-file-populated.test
# RUN: diff %S/Inputs/diff-test-update/single-split-file-no-expected.out %S/Inputs/diff-test-update/single-split-file-no-expected.test
# RUN: diff %S/Inputs/diff-test-update/split-c-comments.out %S/Inputs/diff-test-update/split-c-comments.test
# RUN: diff %S/Inputs/diff-test-update/split-whitespace.out "%S/Inputs/diff-test-update/split whitespace.test"


# CHECK: # update-diff-test: could not deduce source and target from {{.*}}1.in and {{.*}}2.in
# CHECK: # update-diff-test: could not deduce source and target from {{.*}}1.txt and {{.*}}2.txt
# CHECK: # update-diff-test: copied {{.*}}my-file.txt to {{.*}}my-file.expected
# CHECK: # update-diff-test: copied {{.*}}1.txt to {{.*}}empty.txt
# CHECK: # update-diff-test: copied {{.*}}diff-tmp.test.tmp.txt to {{.*}}diff-t-out.txt
# CHECK: # update-diff-test: could not deduce source and target from {{.*}}split-both.expected and {{.*}}split-both.out
# CHECK: # update-diff-test: copied {{.*}}unrelated-split.txt to {{.*}}unrelated-split.expected


# CHECK: Failed: 14 (100.00%)
