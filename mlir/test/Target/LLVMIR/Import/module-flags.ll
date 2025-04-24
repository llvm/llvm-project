; RUN: mlir-translate -import-llvm -split-input-file -verify-diagnostics %s | FileCheck %s

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{i32 4, !"probe-stack", !"inline-asm"}

; CHECK-LABEL: module attributes {{.*}} {
; CHECK: llvm.module_flags [
; CHECK-SAME: #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>,
; CHECK-SAME: #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>,
; CHECK-SAME: #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>,
; CHECK-SAME: #llvm.mlir.module_flag<max, "uwtable", 2 : i32>,
; CHECK-SAME: #llvm.mlir.module_flag<max, "frame-pointer", 1 : i32>,
; CHECK-SAME: #llvm.mlir.module_flag<override, "probe-stack", "inline-asm">]

; // -----
; expected-warning@-2 {{unsupported module flag value: !4 = !{!"foo", i32 1}}}
!10 = !{ i32 1, !"foo", i32 1 }
!11 = !{ i32 4, !"bar", i32 37 }
!12 = !{ i32 2, !"qux", i32 42 }
!13 = !{ i32 3, !"qux", !{ !"foo", i32 1 }}
!llvm.module.flags = !{ !10, !11, !12, !13 }
