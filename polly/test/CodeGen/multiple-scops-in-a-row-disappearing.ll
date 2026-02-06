; RUN: opt %loadNPMPolly -passes=polly -disable-output -polly-debug < %s 2>&1 | FileCheck %s

; optimizing region1 as invalidates region2's SCoP.
; It is not recognized as a SCoP anymore because of aliasing.

; REQUIRES: asserts
; CHECK: SCoP detected but dismissed
; CHECK: SCoP in Region 'region2_entry => region2_exit' disappeared

define void @testcase(ptr %arg, i32 %arg1, ptr %arg2, i64 %arg3, i64 %arg4, i64 %arg5, i64 %arg6, i64 %arg7, ptr %arg8, i64 %arg9) {
bb:
  %i = sext i32 %arg1 to i64
  br label %region1_entry

region1_entry:                                             ; preds = %region2_exit, %bb
  %i11 = phi i64 [ 0, %bb ], [ 0, %region2_exit ]
  br i1 true, label %bb12, label %bb14

bb12:                                             ; preds = %bb12, %region1_entry
  store <2 x i64> zeroinitializer, ptr null, align 16
  %i13 = icmp eq i64 0, %arg3
  br i1 %i13, label %bb14, label %bb12

bb14:                                             ; preds = %bb14, %bb12, %region1_entry
  %i15 = load <8 x i16>, ptr null, align 16
  %i16 = icmp eq i64 0, %arg3
  br i1 %i16, label %region1_exit, label %bb14

region1_exit:                                             ; preds = %bb14
  call void null(ptr null, ptr null, i8 0)
  br i1 false, label %region2_entry, label %region2_exit

region2_entry:                                             ; preds = %region2_entry, %region1_exit
  store <2 x i64> zeroinitializer, ptr null, align 16
  br i1 true, label %bb19, label %region2_entry

bb19:                                             ; preds = %region2_entry
  %i20 = mul i64 %i11, %i
  %i21 = getelementptr i8, ptr %arg, i64 %i20
  %i22 = load i32, ptr null, align 4
  br label %bb24

bb24:                                             ; preds = %bb24, %bb19
  %i25 = load i64, ptr %i21, align 1
  %i26 = getelementptr i8, ptr %i21, i64 %i
  %i27 = load i64, ptr %i26, align 1
  store i64 0, ptr %arg2, align 1
  %i28 = getelementptr i8, ptr %i26, i64 %i
  %i29 = load i64, ptr %arg, align 1
  %i30 = getelementptr i8, ptr %i28, i64 %arg4
  %i31 = getelementptr i8, ptr %i30, i64 %arg9
  %i32 = getelementptr i8, ptr %i31, i64 %arg6
  %i33 = getelementptr i8, ptr %i32, i64 %arg5
  %i34 = getelementptr i8, ptr %i33, i64 %arg7
  %i35 = load i64, ptr %i34, align 1
  %i36 = icmp eq i64 0, %arg3
  br i1 %i36, label %region2_exit, label %bb24

region2_exit:                                             ; preds = %bb24, %region1_exit
  br i1 false, label %bb37, label %region1_entry

bb37:                                             ; preds = %region2_exit
  ret void
}
