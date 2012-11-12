; PR1135
; RUN: llc %s -o - | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.3"


; CHECK: 	movl	%{{.*}},   (%rdi,[[R0:.+]],4)
; CHECK:	movl	%{{.*}},  8(%rdi,[[R0]],4)
; CHECK:	movl	%{{.*}},  4(%rdi,[[R0]],4)
; CHECK:	movl	%{{.*}}, 12(%rdi,[[R0]],4)

define void @test(i32* nocapture %array, i32 %r0) nounwind ssp noredzone {
bb.nph:
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %j.010 = phi i8 [ 0, %bb.nph ], [ %14, %bb ]    ; <i8> [#uses=1]
  %k.19 = phi i8 [ 0, %bb.nph ], [ %.k.1, %bb ]   ; <i8> [#uses=1]
  %i0.08 = phi i8 [ 0, %bb.nph ], [ %15, %bb ]    ; <i8> [#uses=3]
  %0 = icmp slt i8 %i0.08, 4                      ; <i1> [#uses=1]
  %iftmp.0.0 = select i1 %0, i8 %i0.08, i8 0      ; <i8> [#uses=2]
  %1 = icmp eq i8 %i0.08, 4                       ; <i1> [#uses=1]
  %2 = zext i1 %1 to i8                           ; <i8> [#uses=1]
  %.k.1 = add i8 %2, %k.19                        ; <i8> [#uses=2]
  %3 = shl i8 %.k.1, 2                            ; <i8> [#uses=1]
  %4 = add i8 %3, %iftmp.0.0                      ; <i8> [#uses=1]
  %5 = shl i8 %4, 2                               ; <i8> [#uses=1]
  %6 = zext i8 %5 to i64                          ; <i64> [#uses=4]
  %7 = getelementptr inbounds i32* %array, i64 %6 ; <i32*> [#uses=1]
  store i32 %r0, i32* %7, align 4
  %8 = or i64 %6, 2                               ; <i64> [#uses=1]
  %9 = getelementptr inbounds i32* %array, i64 %8 ; <i32*> [#uses=1]
  store i32 %r0, i32* %9, align 4
  %10 = or i64 %6, 1                              ; <i64> [#uses=1]
  %11 = getelementptr inbounds i32* %array, i64 %10 ; <i32*> [#uses=1]
  store i32 %r0, i32* %11, align 4
  %12 = or i64 %6, 3                              ; <i64> [#uses=1]
  %13 = getelementptr inbounds i32* %array, i64 %12 ; <i32*> [#uses=1]
  store i32 %r0, i32* %13, align 4
  %14 = add nsw i8 %j.010, 1                      ; <i8> [#uses=2]
  %15 = add i8 %iftmp.0.0, 1                      ; <i8> [#uses=1]
  %exitcond = icmp eq i8 %14, 32                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb
  ret void
}

; CHECK: test1:
; CHECK: 	movl	%{{.*}},   (%[[RDI:...]],%[[RCX:...]],4)
; CHECK:	movl	%{{.*}},  8(%[[RDI]],%[[RCX]],4)
; CHECK:	movl	%{{.*}},  4(%[[RDI]],%[[RCX]],4)
; CHECK:	movl	%{{.*}}, 12(%[[RDI]],%[[RCX]],4)

define void @test1(i32* nocapture %array, i32 %r0, i8 signext %k, i8 signext %i0) nounwind {
bb.nph:
  br label %for.body

for.body:                                         ; preds = %for.body, %bb.nph
  %j.065 = phi i8 [ 0, %bb.nph ], [ %inc52, %for.body ] ; <i8> [#uses=1]
  %i0.addr.064 = phi i8 [ %i0, %bb.nph ], [ %add, %for.body ] ; <i8> [#uses=3]
  %k.addr.163 = phi i8 [ %k, %bb.nph ], [ %inc.k.addr.1, %for.body ] ; <i8> [#uses=1]
  %cmp5 = icmp slt i8 %i0.addr.064, 4             ; <i1> [#uses=1]
  %cond = select i1 %cmp5, i8 %i0.addr.064, i8 0  ; <i8> [#uses=2]
  %cmp12 = icmp eq i8 %i0.addr.064, 4             ; <i1> [#uses=1]
  %inc = zext i1 %cmp12 to i8                     ; <i8> [#uses=1]
  %inc.k.addr.1 = add i8 %inc, %k.addr.163        ; <i8> [#uses=2]
  %mul = shl i8 %cond, 2                          ; <i8> [#uses=1]
  %mul22 = shl i8 %inc.k.addr.1, 4                ; <i8> [#uses=1]
  %add23 = add i8 %mul22, %mul                    ; <i8> [#uses=1]
  %idxprom = zext i8 %add23 to i64                ; <i64> [#uses=4]
  %arrayidx = getelementptr inbounds i32* %array, i64 %idxprom ; <i32*> [#uses=1]
  store i32 %r0, i32* %arrayidx
  %add3356 = or i64 %idxprom, 2                   ; <i64> [#uses=1]
  %arrayidx36 = getelementptr inbounds i32* %array, i64 %add3356 ; <i32*> [#uses=1]
  store i32 %r0, i32* %arrayidx36
  %add4058 = or i64 %idxprom, 1                   ; <i64> [#uses=1]
  %arrayidx43 = getelementptr inbounds i32* %array, i64 %add4058 ; <i32*> [#uses=1]
  store i32 %r0, i32* %arrayidx43
  %add4760 = or i64 %idxprom, 3                   ; <i64> [#uses=1]
  %arrayidx50 = getelementptr inbounds i32* %array, i64 %add4760 ; <i32*> [#uses=1]
  store i32 %r0, i32* %arrayidx50
  %inc52 = add nsw i8 %j.065, 1                   ; <i8> [#uses=2]
  %add = add i8 %cond, 1                          ; <i8> [#uses=1]
  %exitcond = icmp eq i8 %inc52, 32               ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
