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

; // -----

!llvm.module.flags = !{!31873}

!31873 = !{i32 1, !"ProfileSummary", !31874}
!31874 = !{!31875, !31876, !31877, !31878, !31879, !31880, !31881, !31882, !31883, !31884}
!31875 = !{!"ProfileFormat", !"InstrProf"}
!31876 = !{!"TotalCount", i64 263646}
!31877 = !{!"MaxCount", i64 86427}
!31878 = !{!"MaxInternalCount", i64 86427}
!31879 = !{!"MaxFunctionCount", i64 4691}
!31880 = !{!"NumCounts", i64 3712}
!31881 = !{!"NumFunctions", i64 796}
!31882 = !{!"IsPartialProfile", i64 0}
!31883 = !{!"PartialProfileRatio", double 0.000000e+00}
!31884 = !{!"DetailedSummary", !31885}
!31885 = !{!31886, !31887}
!31886 = !{i32 10000, i64 86427, i32 1}
!31887 = !{i32 100000, i64 86427, i32 1}

; CHECK: llvm.module_flags [#llvm.mlir.module_flag<error, "ProfileSummary",
; CHECK-SAME: #llvm.profile_summary<format = InstrProf, total_count = 263646,
; CHECK-SAME: max_count = 86427, max_internal_count = 86427, max_function_count = 4691,
; CHECK-SAME: num_counts = 3712, num_functions = 796, is_partial_profile = 0,
; CHECK-SAME: partial_profile_ratio = 0.000000e+00 : f64,
; CHECK-SAME: detailed_summary =
; CHECK-SAME: <cut_off = 10000, min_count = 86427, num_counts = 1>,
; CHECK-SAME: <cut_off = 100000, min_count = 86427, num_counts = 1>
; CHECK-SAME: >>]

; // -----

; Test optional fields

!llvm.module.flags = !{!41873}

!41873 = !{i32 1, !"ProfileSummary", !41874}
!41874 = !{!41875, !41876, !41877, !41878, !41879, !41880, !41881, !41884}
!41875 = !{!"ProfileFormat", !"InstrProf"}
!41876 = !{!"TotalCount", i64 263646}
!41877 = !{!"MaxCount", i64 86427}
!41878 = !{!"MaxInternalCount", i64 86427}
!41879 = !{!"MaxFunctionCount", i64 4691}
!41880 = !{!"NumCounts", i64 3712}
!41881 = !{!"NumFunctions", i64 796}
!41884 = !{!"DetailedSummary", !41885}
!41885 = !{!41886, !41887}
!41886 = !{i32 10000, i64 86427, i32 1}
!41887 = !{i32 100000, i64 86427, i32 1}

; CHECK: llvm.module_flags [#llvm.mlir.module_flag<error, "ProfileSummary",
; CHECK-SAME: #llvm.profile_summary<format = InstrProf, total_count = 263646,
; CHECK-SAME: max_count = 86427, max_internal_count = 86427, max_function_count = 4691,
; CHECK-SAME: num_counts = 3712, num_functions = 796,
; CHECK-SAME: detailed_summary =
; CHECK-SAME: <cut_off = 10000, min_count = 86427, num_counts = 1>,
; CHECK-SAME: <cut_off = 100000, min_count = 86427, num_counts = 1>
; CHECK-SAME: >>]
