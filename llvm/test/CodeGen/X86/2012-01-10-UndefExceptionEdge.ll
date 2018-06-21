; RUN: llc < %s -disable-fp-elim
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.7"

; This test case has a landing pad with two predecessors, and a variable that
; is undef on the first edge while carrying the first function return value on
; the second edge.
;
; Live range splitting tries to isolate the block containing the first function
; call, and it is important that the last split point is after the function call
; so the return value can spill.
;
; <rdar://problem/10664933>

@Exception = external unnamed_addr constant { i8*, i8* }

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define void @f(i32* nocapture %arg, i32* nocapture %arg1, i32* nocapture %arg2, i32* nocapture %arg3, i32 %arg4, i32 %arg5) optsize ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  br i1 undef, label %bb6, label %bb7

bb6:                                              ; preds = %bb
  %tmp = select i1 false, i32 0, i32 undef
  br label %bb7

bb7:                                              ; preds = %bb6, %bb
  %tmp8 = phi i32 [ %tmp, %bb6 ], [ 0, %bb ]
  %tmp9 = shl i32 %tmp8, 2
  %tmp10 = invoke noalias i8* @_Znam(i32 undef) optsize
          to label %bb11 unwind label %bb20

bb11:                                             ; preds = %bb7
  %tmp12 = ptrtoint i8* %tmp10 to i32
  %tmp13 = bitcast i8* %tmp10 to i32*
  %tmp14 = shl i32 %tmp8, 2
  %tmp15 = getelementptr i32, i32* %tmp13, i32 undef
  %tmp16 = getelementptr i32, i32* %tmp13, i32 undef
  %tmp17 = zext i32 %tmp9 to i64
  %tmp18 = add i64 %tmp17, -1
  %tmp19 = icmp ugt i64 %tmp18, 4294967295
  br i1 %tmp19, label %bb29, label %bb31

bb20:                                             ; preds = %bb43, %bb41, %bb29, %bb7
  %tmp21 = phi i32 [ undef, %bb7 ], [ %tmp12, %bb43 ], [ %tmp12, %bb29 ], [ %tmp12, %bb41 ]
  %tmp22 = landingpad { i8*, i32 }
          catch i8* bitcast ({ i8*, i8* }* @Exception to i8*)
  br i1 undef, label %bb23, label %bb69

bb23:                                             ; preds = %bb38, %bb20
  %tmp24 = phi i32 [ %tmp12, %bb38 ], [ %tmp21, %bb20 ]
  %tmp25 = icmp eq i32 %tmp24, 0
  br i1 %tmp25, label %bb28, label %bb26

bb26:                                             ; preds = %bb23
  %tmp27 = inttoptr i32 %tmp24 to i8*
  br label %bb28

bb28:                                             ; preds = %bb26, %bb23
  ret void

bb29:                                             ; preds = %bb11
  invoke void @OnOverFlow() optsize
          to label %bb30 unwind label %bb20

bb30:                                             ; preds = %bb29
  unreachable

bb31:                                             ; preds = %bb11
  %tmp32 = bitcast i32* %tmp15 to i8*
  %tmp33 = zext i32 %tmp8 to i64
  %tmp34 = add i64 %tmp33, -1
  %tmp35 = icmp ugt i64 %tmp34, 4294967295
  %tmp36 = icmp sgt i32 %tmp8, 0
  %tmp37 = add i32 %tmp9, -4
  br label %bb38

bb38:                                             ; preds = %bb67, %bb31
  %tmp39 = phi i32 [ %tmp68, %bb67 ], [ undef, %bb31 ]
  %tmp40 = icmp sgt i32 %tmp39, undef
  br i1 %tmp40, label %bb41, label %bb23

bb41:                                             ; preds = %bb38
  invoke void @Pjii(i32* %tmp16, i32 0, i32 %tmp8) optsize
          to label %bb42 unwind label %bb20

bb42:                                             ; preds = %bb41
  tail call void @llvm.memset.p0i8.i32(i8* %tmp32, i8 0, i32 %tmp9, i32 1, i1 false) nounwind
  br i1 %tmp35, label %bb43, label %bb45

bb43:                                             ; preds = %bb42
  invoke void @OnOverFlow() optsize
          to label %bb44 unwind label %bb20

bb44:                                             ; preds = %bb43
  unreachable

bb45:                                             ; preds = %bb57, %bb42
  %tmp46 = phi i32 [ %tmp58, %bb57 ], [ 255, %bb42 ]
  %tmp47 = icmp slt i32 undef, 0
  br i1 %tmp47, label %bb48, label %bb59

bb48:                                             ; preds = %bb45
  tail call void @llvm.memset.p0i8.i32(i8* %tmp32, i8 0, i32 %tmp9, i32 1, i1 false) nounwind
  br i1 %tmp36, label %bb49, label %bb57

bb49:                                             ; preds = %bb49, %bb48
  %tmp50 = phi i32 [ %tmp55, %bb49 ], [ 0, %bb48 ]
  %tmp51 = add i32 %tmp50, undef
  %tmp52 = add i32 %tmp50, undef
  %tmp53 = getelementptr i32, i32* %tmp13, i32 %tmp52
  %tmp54 = load i32, i32* %tmp53, align 4
  %tmp55 = add i32 %tmp50, 1
  %tmp56 = icmp eq i32 %tmp55, %tmp8
  br i1 %tmp56, label %bb57, label %bb49

bb57:                                             ; preds = %bb49, %bb48
  %tmp58 = add i32 %tmp46, -1
  br label %bb45

bb59:                                             ; preds = %bb45
  %tmp60 = ashr i32 %tmp46, 31
  tail call void @llvm.memset.p0i8.i32(i8* null, i8 0, i32 %tmp37, i32 1, i1 false) nounwind
  br i1 %tmp36, label %bb61, label %bb67

bb61:                                             ; preds = %bb61, %bb59
  %tmp62 = phi i32 [ %tmp65, %bb61 ], [ 0, %bb59 ]
  %tmp63 = add i32 %tmp62, %tmp14
  %tmp64 = getelementptr i32, i32* %tmp13, i32 %tmp63
  store i32 0, i32* %tmp64, align 4
  %tmp65 = add i32 %tmp62, 1
  %tmp66 = icmp eq i32 %tmp65, %tmp8
  br i1 %tmp66, label %bb67, label %bb61

bb67:                                             ; preds = %bb61, %bb59
  %tmp68 = add i32 %tmp39, -1
  br label %bb38

bb69:                                             ; preds = %bb20
  resume { i8*, i32 } %tmp22
}

declare i32 @__gxx_personality_v0(...)

declare noalias i8* @_Znam(i32) optsize

declare void @Pjii(i32*, i32, i32) optsize

declare i32 @llvm.eh.typeid.for(i8*) nounwind readnone

declare void @OnOverFlow() noreturn optsize ssp align 2
