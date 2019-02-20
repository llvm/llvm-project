; RUN: opt -indvars -disable-output < %s 

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i8:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128-n32"
target triple = "powerpc-unknown-linux-gnu"

define void @vec_inverse_5_7_vert_loop_copyseparate(i8* %x, i32 %n, i32 %rowbytes) nounwind {
entry:
  %tmp1 = sdiv i32 %n, 3                          ; <i32> [#uses=1]
  %tmp2 = sdiv i32 %rowbytes, 5                   ; <i32> [#uses=2]
  br label %bb49

bb49:                                             ; preds = %bb48, %entry
  %x_addr.0 = phi i8* [ %x, %entry ], [ %tmp481, %bb48 ] ; <i8*> [#uses=2]
  br label %bb10

bb10:                                             ; preds = %bb49
  %tmp326 = mul nsw i32 %tmp1, %tmp2              ; <i32> [#uses=1]
  %tmp351 = getelementptr inbounds i8, i8* %x_addr.0, i32 %tmp326 ; <i8*> [#uses=1]
  br i1 false, label %bb.nph, label %bb48

bb.nph:                                           ; preds = %bb10
  br label %bb23

bb23:                                             ; preds = %bb28, %bb.nph
  %pOriginHi.01 = phi i8* [ %tmp351, %bb.nph ], [ %pOriginHi.0, %bb28 ] ; <i8*> [#uses=2]
  %tmp378 = bitcast i8* %pOriginHi.01 to i8*      ; <i8*> [#uses=1]
  store i8* %tmp378, i8** null
  %tmp385 = getelementptr inbounds i8, i8* %pOriginHi.01, i32 %tmp2 ; <i8*> [#uses=1]
  br label %bb28

bb28:                                             ; preds = %bb23
  %pOriginHi.0 = phi i8* [ %tmp385, %bb23 ]       ; <i8*> [#uses=1]
  br i1 false, label %bb23, label %bb28.bb48_crit_edge

bb28.bb48_crit_edge:                              ; preds = %bb28
  br label %bb48

bb48:                                             ; preds = %bb28.bb48_crit_edge, %bb10
  %tmp481 = getelementptr inbounds i8, i8* %x_addr.0, i32 1 ; <i8*> [#uses=1]
  br label %bb49
}
