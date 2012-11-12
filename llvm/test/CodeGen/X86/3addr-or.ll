; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; rdar://7527734

define i32 @test1(i32 %x) nounwind readnone ssp {
entry:
; CHECK: test1:
; CHECK: leal 3(%rdi), %eax
  %0 = shl i32 %x, 5                              ; <i32> [#uses=1]
  %1 = or i32 %0, 3                               ; <i32> [#uses=1]
  ret i32 %1
}

define i64 @test2(i8 %A, i8 %B) nounwind {
; CHECK: test2:
; CHECK: shrq $4
; CHECK-NOT: movq
; CHECK-NOT: orq
; CHECK: leaq
; CHECK: ret
  %C = zext i8 %A to i64                          ; <i64> [#uses=1]
  %D = shl i64 %C, 4                              ; <i64> [#uses=1]
  %E = and i64 %D, 48                             ; <i64> [#uses=1]
  %F = zext i8 %B to i64                          ; <i64> [#uses=1]
  %G = lshr i64 %F, 4                             ; <i64> [#uses=1]
  %H = or i64 %G, %E                              ; <i64> [#uses=1]
  ret i64 %H
}

;; Test that OR is only emitted as LEA, not as ADD.

define void @test3(i32 %x, i32* %P) nounwind readnone ssp {
entry:
; No reason to emit an add here, should be an or.
; CHECK: test3:
; CHECK: orl $3, %edi
  %0 = shl i32 %x, 5
  %1 = or i32 %0, 3
  store i32 %1, i32* %P
  ret void
}

define i32 @test4(i32 %a, i32 %b) nounwind readnone ssp {
entry:
  %and = and i32 %a, 6
  %and2 = and i32 %b, 16
  %or = or i32 %and2, %and
  ret i32 %or
; CHECK: test4:
; CHECK: leal	(%rsi,%rdi), %eax
}

define void @test5(i32 %a, i32 %b, i32* nocapture %P) nounwind ssp {
entry:
  %and = and i32 %a, 6
  %and2 = and i32 %b, 16
  %or = or i32 %and2, %and
  store i32 %or, i32* %P, align 4
  ret void
; CHECK: test5:
; CHECK: orl
}
