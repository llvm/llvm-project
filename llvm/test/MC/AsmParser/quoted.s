# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -triple x86_64 a.s | FileCheck %s
# RUN: not llvm-mc -triple x86_64 err.s 2>&1 | FileCheck %s --check-prefix=ERR

#--- a.s
# CHECK: .type "a b",@function
# CHECK: "a b":
.type "a b", @function
"a b":
  call "a b"

#--- err.s
 "a\":
# ERR: 1:2: error: unterminated string constant
# ERR: 1:2: error: unexpected token at start of statement
