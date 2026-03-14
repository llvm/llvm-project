; RUN: not llc -o /dev/null -mattr=avx %s 2>&1 | FileCheck %s
target triple = "x86_64--"

; CHECK: error: register 'XMM15' allocated for constraint '{xmm15}' does not match required type
define void @test1() nounwind {
entry:
  tail call void asm sideeffect "call dummy", "{xmm15},~{dirflag},~{fpsr},~{flags}"(<8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>) #1
  ret void
}
