; RUN: mlir-translate -import-llvm -split-input-file -verify-diagnostics %s | FileCheck %s

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}

; CHECK-LABEL: module attributes {{.*}} {
; CHECK: llvm.module_flags [
; CHECK-SAME: #llvm.mlir.module_flag<error, "wchar_size", 4>,
; CHECK-SAME: #llvm.mlir.module_flag<min, "PIC Level", 2>,
; CHECK-SAME: #llvm.mlir.module_flag<max, "PIE Level", 2>,
; CHECK-SAME: #llvm.mlir.module_flag<max, "uwtable", 2>,
; CHECK-SAME: #llvm.mlir.module_flag<max, "frame-pointer", 1>]
; CHECK: }

; // -----

!llvm.module.flags = !{!0}

; expected-warning@-5{{unsupported module flag value: !"yolo_more", only constant integer currently supported}}
!0 = !{i32 1, !"yolo", !"yolo_more"}
