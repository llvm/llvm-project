; RUN: llc < %s -march=thumb | FileCheck %s
define i32 @t1(i32 %x, i32 %y) nounwind {
entry:
  ; CHECK: mov r0, r12
  %0 = tail call i32 asm "mov $0, $1", "=l,h"(i32 %y) nounwind
  ret i32 %0
}
