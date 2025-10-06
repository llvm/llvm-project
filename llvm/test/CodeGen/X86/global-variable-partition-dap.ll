target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Comments for RUN command options
; 1. `-relocation-model=pic` -> `hot_relro_array` and `cold_relro_array` are
;    placed in the .data.rel.ro-prefixed section.
; 2. `-data-sections=true -unique-section-names=false` -> data sections are
;    uniqufied by variable names.
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic \
; RUN:     -partition-static-data-sections=true \
; RUN:     -data-sections=true  -unique-section-names=false \
; RUN:     %s -o - 2>&1 | FileCheck %s --check-prefixes=UNIQ,COMMON --dump-input=always

; For @.str and @.str.1
; COMMON:      .type .L.str,@object
; UNIQ-NEXT:   .section	.rodata.str1.1.hot.,"aMS",@progbits,1
; COMMON-NEXT: .L.str:
; COMMON-NEXT:    "hot\t"
; COMMON:      .L.str.1:
; COMMON-NEXT:    "%d\t%d\t%d\n"

; For @hot_relro_array
; COMMON:      .type hot_relro_array,@object
; UNIQ-NEXT:   .section	.data.rel.ro.hot.,"aw",@progbits,unique,1

; For @external_hot_data, which is accessed by {cold_func, unprofiled_func, hot_func}.
; COMMON:      .type external_hot_data,@object
; UNIQ-NEXT:   .section	.data.hot.,"aw",@progbits,unique,2

; For @hot_bss, which is accessed by {unprofiled_func, hot_func}.
; COMMON:      .type hot_bss,@object
; UNIQ-NEXT:   .section	.bss.hot.,"aw",@nobits,unique,3

; For @.str.2
; COMMON:      .type .L.str.2,@object
; UNIQ-NEXT:   .section	.rodata.str1.1.unlikely.,"aMS",@progbits,1
; COMMON-NEXT: .L.str.2:
; COMMON-NEXT:    "cold%d\t%d\t%d\n"

; For @cold_bss
; COMMON:      .type cold_bss,@object
; UNIQ-NEXT:   .section	.bss.unlikely.,"aw",@nobits,unique,4

; For @cold_data
; COMMON:      .type cold_data,@object
; UNIQ-NEXT:   .section	.data.unlikely.,"aw",@progbits,unique,5

; For @cold_data_custom_foo_section
; It has an explicit section 'foo' and shouldn't have hot or unlikely suffix.
; COMMON:      .type cold_data_custom_foo_section,@object
; UNIQ-NEXT:   .section foo,"aw",@progbits

; For @cold_relro_array
; COMMON:      .type cold_relro_array,@object
; UNIQ-NEXT:   .section	.data.rel.ro.unlikely.,"aw",@progbits,unique,6


; @bss2 and @data3 are indirectly accessed via @hot_relro_array and
; @cold_relro_array, and actually hot due to accesses via @hot_relro_array.
; Under the hood, the static data splitter pass analyzes accesses from code but
; won't aggressively propgate the hotness of @hot_relro_array into the array
; elements -- instead, this pass reconciles the hotness information from both
; global variable section prefix and PGO counters.

; @bss2 has a section prefix 'hot' in the IR. StaticDataProfileInfo reconciles
; it into a hot prefix.
; COMMON:      .type bss2,@object
; UNIQ-NEXT:   .section	.bss.hot.,"aw",@nobits,unique,7

; @data3 doesn't have data access profile coverage and thereby doesn't have a
; section prefix. PGO counter analysis categorizes it as cold, so it will have
; section name `.data.unlikely`.
; COMMON:      .type data3,@object
; UNIQ-NEXT:   .section	.data,"aw",@progbits,unique,8

; For @data_with_unknown_hotness
; COMMON: 	       .type	.Ldata_with_unknown_hotness,@object          # @data_with_unknown_hotness
; UNIQ:        .section  .data,"aw",@progbits,unique,9

; For @hot_data_custom_bar_section
; It has an explicit section attribute 'var' and shouldn't have hot or unlikely suffix.
; COMMON:      .type hot_data_custom_bar_section,@object
; UNIQ:        .section bar,"aw",@progbits

; For @external_cold_bss
; COMMON:      .type external_cold_bss,@object
; UNIQ-NEXT:   .section	.bss,"aw",@nobits,unique,

@.str = private unnamed_addr constant [5 x i8] c"hot\09\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"%d\09%d\09%d\0A\00", align 1
@hot_relro_array = internal constant [2 x ptr] [ptr @bss2, ptr @data3]
@external_hot_data = global i32 5, !section_prefix !17
@hot_bss = internal global i32 0
@.str.2 = private unnamed_addr constant [14 x i8] c"cold%d\09%d\09%d\0A\00", align 1
@cold_bss = internal global i32 0, !section_prefix !18
@cold_data = internal global i32 4, !section_prefix !18
@cold_data_custom_foo_section = internal global i32 100, section "foo"
@cold_relro_array = internal constant [2 x ptr] [ptr @data3, ptr @bss2], !section_prefix !18
@bss2 = internal global i32 0, !section_prefix !17
@data3 = internal global i32 3
@data_with_unknown_hotness = private global i32 5
@hot_data_custom_bar_section = internal global i32 101 #0
@external_cold_bss = global i32 0

define void @cold_func(i32 %0) !prof !15 {
  %2 = load i32, ptr @cold_bss
  %3 = load i32, ptr @cold_data
  %4 = srem i32 %0, 2
  %5 = sext i32 %4 to i64
  %6 = getelementptr inbounds [2 x ptr], ptr @cold_relro_array, i64 0, i64 %5
  %7 = load ptr, ptr %6
  %8 = load i32, ptr %7
  %9 = load i32, ptr @data_with_unknown_hotness
  %11 = load i32, ptr @external_hot_data
  %12 = load i32, ptr @cold_data_custom_foo_section
  %val = load i32, ptr @external_cold_bss
  %13 = call i32 (...) @func_taking_arbitrary_param(ptr @.str.2, i32 %2, i32 %3, i32 %8, i32 %9, i32 %11, i32 %12, i32 %val)
  ret void
}

define i32 @unprofiled_func() {
  %a = load i32, ptr @data_with_unknown_hotness
  %b = load i32, ptr @external_hot_data
  %c = load i32, ptr @hot_bss
  %ret = call i32 (...) @func_taking_arbitrary_param(i32 %a, i32 %b, i32 %c)
  ret i32 %ret
}

define void @hot_func(i32 %0) !prof !14 {
  %2 = call i32 (...) @func_taking_arbitrary_param(ptr @.str)
  %3 = srem i32 %0, 2
  %4 = sext i32 %3 to i64
  %5 = getelementptr inbounds [2 x ptr], ptr @hot_relro_array, i64 0, i64 %4
  %6 = load ptr, ptr %5
  %7 = load i32, ptr %6
  %8 = load i32, ptr @external_hot_data
  %9 = load i32, ptr @hot_bss
  %10 = load i32, ptr @hot_data_custom_bar_section
  %11 = call i32 (...) @func_taking_arbitrary_param(ptr @.str.1, i32 %7, i32 %8, i32 %9, i32 %10)
  ret void
}

declare i32 @func_taking_arbitrary_param(...)

attributes #0 = {"data-section"="bar"}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"HasDataAccessProf", i32 1}
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

