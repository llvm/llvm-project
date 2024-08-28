; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK:   llvm.func @vec_type_hint() attributes {vec_type_hint = #llvm.vec_type_hint<hint = i32>}
declare !vec_type_hint !0 void @vec_type_hint()

; CHECK:   llvm.func @vec_type_hint_signed() attributes {vec_type_hint = #llvm.vec_type_hint<hint = i32, is_signed = true>}
declare !vec_type_hint !1 void @vec_type_hint_signed()

; CHECK:   llvm.func @vec_type_hint_signed_vec() attributes {vec_type_hint = #llvm.vec_type_hint<hint = vector<2xi32>, is_signed = true>}
declare !vec_type_hint !2 void @vec_type_hint_signed_vec()

; CHECK:   llvm.func @vec_type_hint_float_vec() attributes {vec_type_hint = #llvm.vec_type_hint<hint = vector<3xf32>>}
declare !vec_type_hint !3 void @vec_type_hint_float_vec()

; CHECK:   llvm.func @vec_type_hint_bfloat_vec() attributes {vec_type_hint = #llvm.vec_type_hint<hint = vector<8xbf16>>}
declare !vec_type_hint !4 void @vec_type_hint_bfloat_vec()

; CHECK:   llvm.func @work_group_size_hint() attributes {work_group_size_hint = array<i32: 128, 128, 128>}
declare !work_group_size_hint !5 void @work_group_size_hint()

; CHECK:   llvm.func @reqd_work_group_size() attributes {reqd_work_group_size = array<i32: 128, 256, 128>}
declare !reqd_work_group_size !6 void @reqd_work_group_size()

; CHECK:   llvm.func @intel_reqd_sub_group_size() attributes {intel_reqd_sub_group_size = 32 : i32}
declare !intel_reqd_sub_group_size !7 void @intel_reqd_sub_group_size()

!0 = !{i32 undef, i32 0}
!1 = !{i32 undef, i32 1}
!2 = !{<2 x i32> undef, i32 1}
!3 = !{<3 x float> undef, i32 0}
!4 = !{<8 x bfloat> undef, i32 0}
!5 = !{i32 128, i32 128, i32 128}
!6 = !{i32 128, i32 256, i32 128}
!7 = !{i32 32}
