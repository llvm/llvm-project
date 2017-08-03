; RUN: llc -march=x86-64 < %s > /dev/null

; ScalarEvolution misses an opportunity to fold ((trunc x) + (trunc -x) + y),
; but LSR should tolerate this.
; rdar://7886751

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0"

define fastcc void @formatValue(i64 %arg5) nounwind {
bb12:                                             ; preds = %bb11
  %t = trunc i64 %arg5 to i32                   ; <i32> [#uses=1]
  %t13 = sub i64 0, %arg5                       ; <i64> [#uses=1]
  %t14 = and i64 %t13, 4294967295             ; <i64> [#uses=1]
  br label %bb15

bb15:                                             ; preds = %bb21, %bb12
  %t16 = phi i64 [ 0, %bb12 ], [ %t23, %bb15 ] ; <i64> [#uses=2]
  %t17 = mul i64 %t14, %t16                 ; <i64> [#uses=1]
  %t18 = add i64 undef, %t17                  ; <i64> [#uses=1]
  %t19 = trunc i64 %t18 to i32                ; <i32> [#uses=1]
  %t22 = icmp eq i32 %t19, %t               ; <i1> [#uses=1]
  %t23 = add i64 %t16, 1                      ; <i64> [#uses=1]
  br i1 %t22, label %bb24, label %bb15

bb24:                                             ; preds = %bb21, %bb11
  unreachable
}

; ScalarEvolution should be able to correctly expand the crazy addrec here.
; PR6914

define void @int323() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %lbl_264, %for.inc, %entry
  %g_263.tmp.1 = phi i8 [ undef, %entry ], [ %g_263.tmp.1, %for.cond ]
  %p_95.addr.0 = phi i8 [ 0, %entry ], [ %add, %for.cond ]
  %add = add i8 %p_95.addr.0, 1                   ; <i8> [#uses=1]
  br i1 undef, label %for.cond, label %lbl_264

lbl_264:                                          ; preds = %if.end, %lbl_264.preheader
  %g_263.tmp.0 = phi i8 [ %g_263.tmp.1, %for.cond ] ; <i8> [#uses=1]
  %tmp7 = load i16, i16* undef                         ; <i16> [#uses=1]
  %conv8 = trunc i16 %tmp7 to i8                  ; <i8> [#uses=1]
  %mul.i = mul i8 %p_95.addr.0, %p_95.addr.0      ; <i8> [#uses=1]
  %mul.i18 = mul i8 %mul.i, %conv8                ; <i8> [#uses=1]
  %tobool12 = icmp eq i8 %mul.i18, 0              ; <i1> [#uses=1]
  unreachable
}

; LSR ends up going into conservative pruning mode; don't prune the solution
; so far that it becomes unsolvable though.
; PR7077

%struct.Bu = type { i32, i32, i32 }

define void @_Z3fooP2Bui(%struct.Bu* nocapture %bu) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc131, %entry
  %indvar = phi i64 [ %indvar.next, %for.inc131 ], [ 0, %entry ] ; <i64> [#uses=3]
  br i1 undef, label %for.inc131, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %for.body
  %tmp15 = add i64 %indvar, 1                     ; <i64> [#uses=1]
  %tmp17 = add i64 %indvar, 2                      ; <i64> [#uses=1]
  %tmp19 = add i64 %indvar, 3                      ; <i64> [#uses=1]
  %tmp21 = add i64 %indvar, 4                      ; <i64> [#uses=1]
  %tmp23 = add i64 %indvar, 5                      ; <i64> [#uses=1]
  %tmp25 = add i64 %indvar, 6                      ; <i64> [#uses=1]
  %tmp27 = add i64 %indvar, 7                      ; <i64> [#uses=1]
  %tmp29 = add i64 %indvar, 8                      ; <i64> [#uses=1]
  %tmp31 = add i64 %indvar, 9                      ; <i64> [#uses=1]
  %tmp35 = add i64 %indvar, 11                     ; <i64> [#uses=1]
  %tmp37 = add i64 %indvar, 12                     ; <i64> [#uses=1]
  %tmp39 = add i64 %indvar, 13                     ; <i64> [#uses=1]
  %tmp41 = add i64 %indvar, 14                     ; <i64> [#uses=1]
  %tmp43 = add i64 %indvar, 15                     ; <i64> [#uses=1]
  %tmp45 = add i64 %indvar, 16                     ; <i64> [#uses=1]
  %tmp47 = add i64 %indvar, 17                     ; <i64> [#uses=1]
  %mul = trunc i64 %indvar to i32                  ; <i32> [#uses=1]
  %add22 = trunc i64 %tmp15 to i32                ; <i32> [#uses=1]
  %add28 = trunc i64 %tmp17 to i32                ; <i32> [#uses=1]
  %add34 = trunc i64 %tmp19 to i32                ; <i32> [#uses=1]
  %add40 = trunc i64 %tmp21 to i32                ; <i32> [#uses=1]
  %add46 = trunc i64 %tmp23 to i32                ; <i32> [#uses=1]
  %add52 = trunc i64 %tmp25 to i32                ; <i32> [#uses=1]
  %add58 = trunc i64 %tmp27 to i32                ; <i32> [#uses=1]
  %add64 = trunc i64 %tmp29 to i32                ; <i32> [#uses=1]
  %add70 = trunc i64 %tmp31 to i32                ; <i32> [#uses=1]
  %add82 = trunc i64 %tmp35 to i32                ; <i32> [#uses=1]
  %add88 = trunc i64 %tmp37 to i32                ; <i32> [#uses=1]
  %add94 = trunc i64 %tmp39 to i32                ; <i32> [#uses=1]
  %add100 = trunc i64 %tmp41 to i32               ; <i32> [#uses=1]
  %add106 = trunc i64 %tmp43 to i32               ; <i32> [#uses=1]
  %add112 = trunc i64 %tmp45 to i32               ; <i32> [#uses=1]
  %add118 = trunc i64 %tmp47 to i32               ; <i32> [#uses=1]
  %tmp10 = getelementptr %struct.Bu, %struct.Bu* %bu, i64 %indvar, i32 2 ; <i32*> [#uses=1]
  %tmp11 = load i32, i32* %tmp10                       ; <i32> [#uses=0]
  tail call void undef(i32 %add22)
  tail call void undef(i32 %add28)
  tail call void undef(i32 %add34)
  tail call void undef(i32 %add40)
  tail call void undef(i32 %add46)
  tail call void undef(i32 %add52)
  tail call void undef(i32 %add58)
  tail call void undef(i32 %add64)
  tail call void undef(i32 %add70)
  tail call void undef(i32 %add82)
  tail call void undef(i32 %add88)
  tail call void undef(i32 %add94)
  tail call void undef(i32 %add100)
  tail call void undef(i32 %add106)
  tail call void undef(i32 %add112)
  tail call void undef(i32 %add118)
  br label %for.body123

for.body123:                                      ; preds = %for.body123, %lor.lhs.false
  %j.03 = phi i32 [ 0, %lor.lhs.false ], [ %inc, %for.body123 ] ; <i32> [#uses=2]
  %add129 = add i32 %mul, %j.03                   ; <i32> [#uses=1]
  tail call void undef(i32 %add129)
  %inc = add nsw i32 %j.03, 1                     ; <i32> [#uses=1]
  br i1 undef, label %for.inc131, label %for.body123

for.inc131:                                       ; preds = %for.body123, %for.body
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br i1 undef, label %for.end134, label %for.body

for.end134:                                       ; preds = %for.inc131
  ret void
}

; LSR needs to remember inserted instructions even in postinc mode, because
; there could be multiple subexpressions within a single expansion which
; require insert point adjustment.
; PR7306

define fastcc i32 @GetOptimum() nounwind {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %t = phi i32 [ 0, %bb ], [ %t2, %bb1 ]      ; <i32> [#uses=1]
  %t2 = add i32 %t, undef                     ; <i32> [#uses=3]
  br i1 undef, label %bb1, label %bb3

bb3:                                              ; preds = %bb1
  %t4 = add i32 undef, -1                       ; <i32> [#uses=1]
  br label %bb5

bb5:                                              ; preds = %bb16, %bb3
  %t6 = phi i32 [ %t17, %bb16 ], [ 0, %bb3 ]  ; <i32> [#uses=3]
  %t7 = add i32 undef, %t6                    ; <i32> [#uses=2]
  %t8 = add i32 %t4, %t6                    ; <i32> [#uses=1]
  br i1 undef, label %bb9, label %bb10

bb9:                                              ; preds = %bb5
  br label %bb10

bb10:                                             ; preds = %bb9, %bb5
  br i1 undef, label %bb11, label %bb16

bb11:                                             ; preds = %bb10
  %t12 = icmp ugt i32 %t7, %t2              ; <i1> [#uses=1]
  %t13 = select i1 %t12, i32 %t2, i32 %t7 ; <i32> [#uses=1]
  br label %bb14

bb14:                                             ; preds = %bb11
  store i32 %t13, i32* null
  ret i32 %t8

bb16:                                             ; preds = %bb10
  %t17 = add i32 %t6, 1                       ; <i32> [#uses=1]
  br label %bb5
}
