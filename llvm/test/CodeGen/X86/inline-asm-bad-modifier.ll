; RUN: not llc -mtriple=x86_64-- < %s 2>&1 | FileCheck %s

;CHECK: error: invalid operand in inline asm: 'vmovd ${1:x}, $0'
define i32 @foo() {
entry:
  %0 = tail call i32 asm sideeffect "vmovd ${1:x}, $0", "=r,x,~{dirflag},~{fpsr},~{flags}"(<2 x i64> <i64 240518168632, i64 240518168632>)
  ret i32 %0
}
