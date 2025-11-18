target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Requires asserts for -debug-only.
; REQUIRES: asserts

; RUN: rm -rf %t && split-file %s %t && cd %t

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic \
; RUN:     -partition-static-data-sections=true \
; RUN:     -debug-only=static-data-profile-info \
; RUN:     -data-sections=true  -unique-section-names=false \
; RUN:     input-with-data-access-prof-on.ll -o - 2>&1 | FileCheck %s --check-prefixes=LOG,IR

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic \
; RUN:     -partition-static-data-sections=true \
; RUN:     -debug-only=static-data-profile-info \
; RUN:     -data-sections=true  -unique-section-names=false \
; RUN:     input-with-data-access-prof-off.ll -o - 2>&1 | FileCheck %s --check-prefixes=OFF

; LOG: hot_bss has section prefix hot, the max from data access profiles as hot and PGO counters as hot
; LOG: data_unknown_hotness has section prefix <empty>, the max from data access profiles as <empty> and PGO counters as unlikely
; LOG: external_relro_array has section prefix unlikely, solely from data access profiles

; IR:          .type   hot_bss,@object
; IR-NEXT:     .section .bss.hot.,"aw"
; IR:          .type   data_unknown_hotness,@object
; IR-NEXT:    .section .data,"aw"
; IR:          .type   external_relro_array,@object
; IR-NEXT:     .section        .data.rel.ro.unlikely.,"aw"


; OFF:        .type   hot_bss,@object
; OFF-NEXT:   .section        .bss.hot.,"aw"
; OFF:        .type   data_unknown_hotness,@object
; OFF-NEXT:   .section        .data.unlikely.,"aw"
;; Global variable section prefix metadata is not used when
;; module flag `EnableDataAccessProf` is 0, and @external_relro_array has
;; external linkage, so analysis based on PGO counters doesn't apply. 
; OFF:        .type   external_relro_array,@object    # @external_relro_array
; OFF-NEXT:   .section        .data.rel.ro,"aw"

;--- input-with-data-access-prof-on.ll
; Internal vars
@hot_bss = internal global i32 0, !section_prefix !17
@data_unknown_hotness = internal global i32 1
; External vars
@external_relro_array = constant [2 x ptr] [ptr @hot_bss, ptr @data_unknown_hotness], !section_prefix !18

define void @cold_func() !prof !15 {
  %9 = load i32, ptr @data_unknown_hotness
  %11 = call i32 (...) @func_taking_arbitrary_param(i32 %9)
  ret void
}

define void @hot_func() !prof !14 {
  %9 = load i32, ptr @hot_bss
  %11 = call i32 (...) @func_taking_arbitrary_param(i32 %9)
  ret void
}

declare i32 @func_taking_arbitrary_param(...)

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"EnableDataAccessProf", i32 1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1460183}
!5 = !{!"MaxCount", i64 849024}
!6 = !{!"MaxInternalCount", i64 32769}
!7 = !{!"MaxFunctionCount", i64 849024}
!8 = !{!"NumCounts", i64 23627}
!9 = !{!"NumFunctions", i64 3271}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13}
!12 = !{i32 990000, i64 166, i32 73}
!13 = !{i32 999999, i64 3, i32 1443}
!14 = !{!"function_entry_count", i64 100000}
!15 = !{!"function_entry_count", i64 1}
!16 = !{!"branch_weights", i32 1, i32 99999}
!17 = !{!"section_prefix", !"hot"}
!18 = !{!"section_prefix", !"unlikely"}

;--- input-with-data-access-prof-off.ll
; Same as file above except that module flag `EnableDataAccessProf` has value 0.
; Internal vars
@hot_bss = internal global i32 0, !section_prefix !17
@data_unknown_hotness = internal global i32 1
; External vars
@external_relro_array = constant [2 x ptr] [ptr @hot_bss, ptr @data_unknown_hotness], !section_prefix !18

define void @cold_func() !prof !15 {
  %9 = load i32, ptr @data_unknown_hotness
  %11 = call i32 (...) @func_taking_arbitrary_param(i32 %9)
  ret void
}

define void @hot_func() !prof !14 {
  %9 = load i32, ptr @hot_bss
  %11 = call i32 (...) @func_taking_arbitrary_param(i32 %9)
  ret void
}

declare i32 @func_taking_arbitrary_param(...)

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"EnableDataAccessProf", i32 0}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1460183}
!5 = !{!"MaxCount", i64 849024}
!6 = !{!"MaxInternalCount", i64 32769}
!7 = !{!"MaxFunctionCount", i64 849024}
!8 = !{!"NumCounts", i64 23627}
!9 = !{!"NumFunctions", i64 3271}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13}
!12 = !{i32 990000, i64 166, i32 73}
!13 = !{i32 999999, i64 3, i32 1443}
!14 = !{!"function_entry_count", i64 100000}
!15 = !{!"function_entry_count", i64 1}
!16 = !{!"branch_weights", i32 1, i32 99999}
!17 = !{!"section_prefix", !"hot"}
!18 = !{!"section_prefix", !"unlikely"}
