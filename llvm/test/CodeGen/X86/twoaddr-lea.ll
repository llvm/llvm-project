;; X's live range extends beyond the shift, so the register allocator
;; cannot coalesce it with Y.  Because of this, a copy needs to be
;; emitted before the shift to save the register value before it is
;; clobbered.  However, this copy is not needed if the register
;; allocator turns the shift into an LEA.  This also occurs for ADD.

; Check that the shift gets turned into an LEA.
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-apple-darwin | FileCheck %s

@G = external global i32

define i32 @test1(i32 %X) nounwind {
; CHECK-LABEL: test1:
; CHECK-NOT: mov
; CHECK: leal 1(%rdi)
        %Z = add i32 %X, 1
        store volatile i32 %Z, i32* @G
        ret i32 %X
}

; rdar://8977508
; The second add should not be transformed to leal nor should it be
; commutted (which would require inserting a copy).
define i32 @test2(i32 inreg %a, i32 inreg %b, i32 %c, i32 %d) nounwind {
entry:
; CHECK-LABEL: test2:
; CHECK: leal
; CHECK-NEXT: addl
; CHECK-NEXT: addl
; CHECK-NEXT: ret
 %add = add i32 %b, %a
 %add3 = add i32 %add, %c
 %add5 = add i32 %add3, %d
 ret i32 %add5
}

; rdar://9002648
define i64 @test3(i64 %x) nounwind readnone ssp {
entry:
; CHECK-LABEL: test3:
; CHECK: leaq (%rdi,%rdi), %rax
; CHECK-NOT: addq
; CHECK-NEXT: ret
  %0 = shl i64 %x, 1
  ret i64 %0
}

@global = external global i32, align 4
@global2 = external global i64, align 8

; Test that liveness is properly updated and we do not encounter the
; assert/crash from http://llvm.org/PR28301
; CHECK-LABEL: ham
define void @ham() {
bb:
  br label %bb1

bb1:
  %tmp = phi i64 [ %tmp40, %bb9 ], [ 0, %bb ]
  %tmp2 = phi i32 [ %tmp39, %bb9 ], [ 0, %bb ]
  %tmp3 = icmp sgt i32 undef, 10
  br i1 %tmp3, label %bb2, label %bb3

bb2:
  %tmp6 = load i32, i32* @global, align 4
  %tmp8 = add nsw i32 %tmp6, %tmp2
  %tmp9 = sext i32 %tmp8 to i64
  br label %bb6

bb3:
; CHECK: subl %e[[REG0:[a-z0-9]+]],
; CHECK: leaq 4({{%[a-z0-9]+}}), %r[[REG0]]
  %tmp14 = phi i64 [ %tmp15, %bb5 ], [ 0, %bb1 ]
  %tmp15 = add nuw i64 %tmp14, 4
  %tmp16 = trunc i64 %tmp14 to i32
  %tmp17 = sub i32 %tmp2, %tmp16
  br label %bb4

bb4:
  %tmp20 = phi i64 [ %tmp14, %bb3 ], [ %tmp34, %bb5 ]
  %tmp28 = icmp eq i32 %tmp17, 0
  br i1 %tmp28, label %bb5, label %bb8

bb5:
  %tmp34 = add nuw nsw i64 %tmp20, 1
  %tmp35 = icmp slt i64 %tmp34, %tmp15
  br i1 %tmp35, label %bb4, label %bb3

bb6:
  store volatile i64 %tmp, i64* @global2, align 8
  store volatile i64 %tmp9, i64* @global2, align 8
  store volatile i32 %tmp6, i32* @global, align 4
  %tmp45 = icmp slt i32 undef, undef
  br i1 %tmp45, label %bb6, label %bb9

bb8:
  unreachable

bb9:
  %tmp39 = add nuw nsw i32 %tmp2, 4
  %tmp40 = add nuw i64 %tmp, 4
  br label %bb1
}
