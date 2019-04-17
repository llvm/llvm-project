; RUN: opt < %s -indvars -S -disable-output

; PR5758
define zeroext i1 @foo() nounwind {
entry:
  indirectbr i8* undef, [label %"202", label %"133"]

"132":                                            ; preds = %"133"
  %0 = add i32 %1, 1                              ; <i32> [#uses=1]
  br label %"133"

"133":                                            ; preds = %"132", %entry
  %1 = phi i32 [ %0, %"132" ], [ 0, %entry ]      ; <i32> [#uses=2]
  %2 = icmp eq i32 %1, 4                          ; <i1> [#uses=1]
  br i1 %2, label %"134", label %"132"

"134":                                            ; preds = %"133"
  ret i1 true

"202":                                            ; preds = %entry
  ret i1 false
}

; PR7333
define void @__atomvec_module__put_vrml_bonds() nounwind {
bb7.preheader:                                    ; preds = %entry
  indirectbr i8* undef, [label %bb14, label %bb16]

bb14:                                             ; preds = %bb14, %bb7.preheader
  br label %bb16

bb16:                                             ; preds = %bb16, %bb14, %bb7.preheader
  %S.31.0 = phi i64 [ %3, %bb16 ], [ 1, %bb7.preheader ], [ 1, %bb14 ] ; <i64> [#uses=2]
  %0 = add nsw i64 %S.31.0, -1                    ; <i64> [#uses=1]
  %1 = getelementptr inbounds [3 x double], [3 x double]* undef, i64 0, i64 %0 ; <double*> [#uses=1]
  %2 = load double, double* %1, align 8                   ; <double> [#uses=0]
  %3 = add nsw i64 %S.31.0, 1                     ; <i64> [#uses=1]
  br label %bb16
}
