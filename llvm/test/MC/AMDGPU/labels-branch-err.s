//  RUN: not llvm-mc -arch=amdgcn -filetype=obj -o /dev/null %s 2>&1 | FileCheck -check-prefix=ERROR %s
//  ERROR: error: undefined label 'undef_label'

s_branch undef_label
