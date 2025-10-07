target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Comments for RUN command options
; 1. `-relocation-model=pic` -> `relro_var` is 
;    placed in the .data.rel.ro-prefixed section.
; 2. `-data-sections=true -unique-section-names=false` -> data sections are
;    uniqufied by variable names.
;
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic \
; RUN:     -partition-static-data-sections=true \
; RUN:     -data-sections=true  -unique-section-names=false \
; RUN:     %s -o - 2>&1 | FileCheck %s  --dump-input=always

; For @.str and @.str.1
; CHECK:      .type .L.str,@object
; CHECK-NEXT:   .section	.rodata.str1.1.hot.,"aMS",@progbits,1
; CHECK-NEXT: .L.str:
; CHECK-NEXT:    "1234"
; CHECK:      .type .str.1,@object
; CHECK:      .str.1:
; CHECK-NEXT:    "abcde"

; For @.str.2
; CHECK:      .type .str.2,@object
; CHECK-NEXT:   .section	.rodata.str1.1,"aMS",@progbits
; CHECK-NEXT:  .globl .str.2
; CHECK-NEXT: .str.2:
; CHECK-NEXT:    "beef"
@.str = private unnamed_addr constant [5 x i8] c"1234\00", align 1
@.str.1 = internal unnamed_addr constant [6 x i8] c"abcde\00"
@.str.2 = unnamed_addr constant [5 x i8] c"beef\00", align 1

; CHECK:      .type relro_var,@object
; CHECK-NEXT:   .section	.data.rel.ro,"aw",@progbits,unique,1

; CHECK:      .type external_hot_data,@object
; CHECK-NEXT:   .section	.data.hot.,"aw",@progbits,unique,2

; CHECK:      .type hot_bss,@object
; CHECK-NEXT:   .section	.bss.hot.,"aw",@nobits,unique,3

@relro_var = constant [2 x ptr] [ptr @bss2, ptr @data3]
@external_hot_data = global i32 5, !section_prefix !17
@hot_bss = internal global i32 0

;; Both section prefix and PGO counters indicate @cold_bss and @cold_data are
;; rarely accesed.
; CHECK:      .type cold_bss,@object
; CHECK-NEXT:   .section	.bss.unlikely.,"aw",@nobits,unique,4
; CHECK:      .type cold_data,@object
; CHECK-NEXT:   .section	.data.unlikely.,"aw",@progbits,unique,5
@cold_bss = internal global i32 0, !section_prefix !18
@cold_data = internal global i32 4, !section_prefix !18

;; @bss2 has a section prefix 'hot' in the IR. StaticDataProfileInfo reconciles
;; it into a hot prefix.
; CHECK:      .type bss2,@object
; CHECK-NEXT:   .section	.bss.hot.,"aw",@nobits,unique,6
@bss2 = internal global i32 0, !section_prefix !17

;; Since `HasDataAccessProf` is true, data without a section prefix is
;; conservatively categorized as unknown (e.g., from incremental source code)
;; rather than cold.
; CHECK:      .type data3,@object
; CHECK-NEXT:   .section	.data,"aw",@progbits,unique,7
@data3 = internal global i32 3

;; These sections have custom names, so they won't be labeled as .hot or .unlikely.
; CHECK:      .type hot_data_custom_bar_section,@object
; CHECK-NEXT:    .section bar,"aw"
; CHECK:      .type cold_data_custom_foo_section,@object
; CHECK-NEXT:   .section foo,"aw"
@hot_data_custom_bar_section = internal global i32 101 #0
@cold_data_custom_foo_section = internal global i32 100, section "foo"

define void @cold_func(i32 %0) !prof !15 {
  %2 = load i32, ptr @cold_bss
  %3 = load i32, ptr @cold_data
  %11 = load i32, ptr @external_hot_data
  %12 = load i32, ptr @cold_data_custom_foo_section
  %13 = call i32 (...) @func_taking_arbitrary_param(ptr @.str.2, i32 %2, i32 %3, i32 %11, i32 %12)
  ret void
}

define i32 @unprofiled_func() {
  %b = load i32, ptr @external_hot_data
  %c = load i32, ptr @hot_bss
  %ret = call i32 (...) @func_taking_arbitrary_param(i32 %b, i32 %c)
  ret i32 %ret
}

define void @hot_func(i32 %0) !prof !14 {
  %2 = call i32 (...) @func_taking_arbitrary_param(ptr @.str)
  %3 = srem i32 %0, 2
  %4 = sext i32 %3 to i64
  %5 = getelementptr inbounds [2 x ptr], ptr @relro_var, i64 0, i64 %4
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
