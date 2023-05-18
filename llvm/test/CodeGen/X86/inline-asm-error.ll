; RUN: not llc -mtriple=i686-- -regalloc=fast -optimize-regalloc=0 < %s 2> %t1
; RUN: not llc -mtriple=i686-- -regalloc=basic      < %s 2> %t2
; RUN: not llc -mtriple=i686-- -regalloc=greedy     < %s 2> %t3
; RUN: FileCheck %s < %t1
; RUN: FileCheck %s < %t2
; RUN: FileCheck %s < %t3

; The register allocator must fail on this function.
; CHECK: error: inline assembly requires more registers than available

define void @f(i32 %x0, i32 %x1, i32 %x2, i32 %x3, i32 %x4, i32 %x5, i32 %x6, i32 %x7, i32 %x8, i32 %x9) nounwind ssp {
entry:
  tail call void asm sideeffect "hello world", "r,r,r,r,r,r,r,r,r,r,~{dirflag},~{fpsr},~{flags}"(i32 %x0, i32 %x1, i32 %x2, i32 %x3, i32 %x4, i32 %x5, i32 %x6, i32 %x7, i32 %x8, i32 %x9) nounwind
  ret void
}
