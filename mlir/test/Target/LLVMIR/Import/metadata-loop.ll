; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.metadata @__llvm_global_metadata {
; CHECK:   llvm.access_group @[[$GROUP0:.*]]
; CHECK:   llvm.access_group @[[$GROUP1:.*]]
; CHECK:   llvm.access_group @[[$GROUP2:.*]]
; CHECK:   llvm.access_group @[[$GROUP3:.*]]
; CHECK: }

; CHECK-LABEL: llvm.func @access_group
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define void @access_group(ptr %arg1) {
  ; CHECK: llvm.load %[[ARG1]] {access_groups = [@__llvm_global_metadata::@[[$GROUP0]], @__llvm_global_metadata::@[[$GROUP1]]]}
  %1 = load i32, ptr %arg1, !llvm.access.group !0
  ; CHECK: llvm.load %[[ARG1]] {access_groups = [@__llvm_global_metadata::@[[$GROUP2]], @__llvm_global_metadata::@[[$GROUP0]]]}
  %2 = load i32, ptr %arg1, !llvm.access.group !1
  ; CHECK: llvm.load %[[ARG1]] {access_groups = [@__llvm_global_metadata::@[[$GROUP3]]]}
  %3 = load i32, ptr %arg1, !llvm.access.group !2
  ret void
}

!0 = !{!3, !4}
!1 = !{!5, !3}
!2 = distinct !{}
!3 = distinct !{}
!4 = distinct !{}
!5 = distinct !{}
