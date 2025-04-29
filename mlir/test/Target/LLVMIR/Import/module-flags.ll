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
; expected-warning@-2 {{unsupported module flag value for key 'qux' : !4 = !{!"foo", i32 1}}}
!10 = !{ i32 1, !"foo", i32 1 }
!11 = !{ i32 4, !"bar", i32 37 }
!12 = !{ i32 2, !"qux", i32 42 }
!13 = !{ i32 3, !"qux", !{ !"foo", i32 1 }}
!llvm.module.flags = !{ !10, !11, !12, !13 }

; // -----

declare void @from(i32)
declare void @to()

!llvm.module.flags = !{!20}

!20 = !{i32 5, !"CG Profile", !21}
!21 = distinct !{!22, !23, !24}
!22 = !{ptr @from, ptr @to, i64 222}
!23 = !{ptr @from, null, i64 222}
!24 = !{ptr @to, ptr @from, i64 222}

; CHECK: llvm.module_flags [#llvm.mlir.module_flag<append, "CG Profile", [
; CHECK-SAME: #llvm.cgprofile_entry<from = @from, to = @to, count = 222>,
; CHECK-SAME: #llvm.cgprofile_entry<from = @from, count = 222>,
; CHECK-SAME: #llvm.cgprofile_entry<from = @to, to = @from, count = 222>
; CHECK-SAME: ]>]
