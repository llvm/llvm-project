; Test linking two functions with different prototypes and two globals
; in different modules.
; RUN: not llvm-link %s %s -o %t.bc 2>&1 | FileCheck %s
; RUN: not llvm-link %s %S/Inputs/redefinition.ll -o %t.bc 2>&1 | FileCheck %s
; CHECK: ERROR: Linking globals named 'foo': symbol multiply defined!
define void @foo() { ret void }
