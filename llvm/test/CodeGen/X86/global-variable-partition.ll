; The static-data-splitter processes data from @cold_func first,
; @unprofiled_func secondly, and @hot_func after the two functions above.
; Tests that data hotness is based on aggregated module-wide profile
; information. This way linker-mergable data is emitted once per module.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The three RUN commands set `-relocation-model=pic` so `hot_relro_array` and
; `cold_relro_array` are placed in the .data.rel.ro-prefixed section.

; This RUN command sets `-data-sections=true -unique-section-names=true` so data
; sections are uniqufied by numbers.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic \
; RUN:     -partition-static-data-sections=true \
; RUN:     -data-sections=true -unique-section-names=true \
; RUN:     %s -o - 2>&1 | FileCheck %s --check-prefixes=SYM,COMMON --dump-input=always

; This RUN command sets `-data-sections=true -unique-section-names=false` so
; data sections are uniqufied by variable names.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic \
; RUN:     -partition-static-data-sections=true \
; RUN:     -data-sections=true  -unique-section-names=false \
; RUN:     %s -o - 2>&1 | FileCheck %s --check-prefixes=UNIQ,COMMON --dump-input=always

; This RUN command sets `-data-sections=false -unique-section-names=false`.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic \
; RUN:     -partition-static-data-sections=true \
; RUN:     -data-sections=false -unique-section-names=false  \
; RUN:     %s -o - 2>&1 | FileCheck %s --check-prefixes=AGG,COMMON --dump-input=always

; For @.str and @.str.1
; COMMON:      .type .L.str,@object
; SYM-NEXT:    .section .rodata.str1.1.hot.
; UNIQ-NEXT:   .section	.rodata.str1.1.hot.,"aMS",@progbits,1
; AGG-NEXT:    .section	.rodata.str1.1.hot
; COMMON-NEXT: .L.str:
; COMMON-NEXT:    "hot\t"
; COMMON:      .L.str.1:
; COMMON-NEXT:    "%d\t%d\t%d\n"

; For @hot_relro_array
; COMMON:      .type hot_relro_array,@object
; SYM-NEXT:    .section	.data.rel.ro.hot.hot_relro_array
; UNIQ-NEXT:   .section	.data.rel.ro.hot.,"aw",@progbits,unique,1
; AGG-NEXT:    .section	.data.rel.ro.hot.,"aw",@progbits

; For @hot_data, which is accessed by {cold_func, unprofiled_func, hot_func}.
; COMMON:      .type hot_data,@object
; SYM-NEXT:    .section	.data.hot.hot_data,"aw",@progbits
; UNIQ-NEXT:   .section	.data.hot.,"aw",@progbits,unique,2
; AGG-NEXT:    .section	.data.hot.,"aw",@progbits

; For @hot_bss, which is accessed by {unprofiled_func, hot_func}.
; COMMON:      .type hot_bss,@object
; SYM-NEXT:    .section	.bss.hot.hot_bss,"aw",@nobits
; UNIQ-NEXT:   .section	.bss.hot.,"aw",@nobits,unique,3
; AGG-NEXT:    .section .bss.hot.,"aw",@nobits

; For @.str.2
; COMMON:      .type .L.str.2,@object
; SYM-NEXT:    .section	.rodata.str1.1.unlikely.,"aMS",@progbits,1
; UNIQ-NEXT:   .section	.rodata.str1.1.unlikely.,"aMS",@progbits,1
; AGG-NEXT:    .section	.rodata.str1.1.unlikely.,"aMS",@progbits,1
; COMMON-NEXT: .L.str.2:
; COMMON-NEXT:    "cold%d\t%d\t%d\n"

; For @cold_bss
; COMMON:      .type cold_bss,@object
; SYM-NEXT:    .section	.bss.unlikely.cold_bss,"aw",@nobits
; UNIQ-NEXT:   .section	.bss.unlikely.,"aw",@nobits,unique,4
; AGG-NEXT:    .section	.bss.unlikely.,"aw",@nobits

; For @cold_data
; COMMON:      .type cold_data,@object
; SYM-NEXT:    .section	.data.unlikely.cold_data,"aw",@progbits
; UNIQ-NEXT:   .section	.data.unlikely.,"aw",@progbits,unique,5
; AGG-NEXT:    .section	.data.unlikely.,"aw",@progbits

; For @cold_data_custom_foo_section
; It has an explicit section 'foo' and shouldn't have hot or unlikely suffix.
; COMMON:      .type cold_data_custom_foo_section,@object
; SYM-NEXT:    .section foo,"aw",@progbits
; UNIQ-NEXT:   .section foo,"aw",@progbits
; AGG-NEXT:    .section foo,"aw",@progbits

; For @cold_relro_array
; COMMON:      .type cold_relro_array,@object
; SYM-NEXT:    .section	.data.rel.ro.unlikely.cold_relro_array,"aw",@progbits
; UNIQ-NEXT:   .section	.data.rel.ro.unlikely.,"aw",@progbits,unique,6
; AGG-NEXT:    .section	.data.rel.ro.unlikely.,"aw",@progbits

; Currently static-data-splitter only analyzes access from code.
; @bss2 and @data3 are indirectly accessed by code through @hot_relro_array
; and @cold_relro_array. A follow-up item is to analyze indirect access via data
; and prune the unlikely list.
; For @bss2
; COMMON:      .type bss2,@object
; SYM-NEXT:    .section	.bss.unlikely.bss2,"aw",@nobits
; UNIQ-NEXT:   .section	.bss.unlikely.,"aw",@nobits,unique,7
; AGG-NEXT:    .section	.bss.unlikely.,"aw",@nobits

; For @data3
; COMMON:      .type data3,@object
; SYM-NEXT:    .section	.data.unlikely.data3,"aw",@progbits
; UNIQ-NEXT:   .section	.data.unlikely.,"aw",@progbits,unique,8
; AGG-NEXT:    .section	.data.unlikely.,"aw",@progbits

;; The `.section` directive is omitted for .data with -unique-section-names=false.
; See MCSectionELF::shouldOmitSectionDirective for the implementation details.

; For @data_with_unknown_hotness
; SYM: 	       .type	.Ldata_with_unknown_hotness,@object          # @data_with_unknown_hotness
; SYM:         .section .data..Ldata_with_unknown_hotness,"aw",@progbits
; UNIQ:        .section  .data,"aw",@progbits,unique,9

; AGG:         .data
; COMMON:      .Ldata_with_unknown_hotness:

; For variables that are not eligible for section prefix annotation
; COMMON:      .type hot_data_custom_bar_section,@object
; SYM-NEXT:    .section bar,"aw",@progbits
; SYM:         hot_data_custom_bar_section
; UNIQ:        .section bar,"aw",@progbits
; AGG:         .section bar,"aw",@progbits

; SYM:      .section .data.llvm.fake_var,"aw"
; UNIQ:     .section .data,"aw"
; AGG:      .data

;; No section for linker declaration
; COMMON-NOT:  qux

@.str = private unnamed_addr constant [5 x i8] c"hot\09\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"%d\09%d\09%d\0A\00", align 1
@hot_relro_array = internal constant [2 x ptr] [ptr @bss2, ptr @data3]
@hot_data = internal global i32 5
@hot_bss = internal global i32 0
@.str.2 = private unnamed_addr constant [14 x i8] c"cold%d\09%d\09%d\0A\00", align 1
@cold_bss = internal global i32 0
@cold_data = internal global i32 4
@cold_data_custom_foo_section = internal global i32 100, section "foo"
@cold_relro_array = internal constant [2 x ptr] [ptr @data3, ptr @bss2]
@bss2 = internal global i32 0
@data3 = internal global i32 3
@data_with_unknown_hotness = private global i32 5
@hot_data_custom_bar_section = internal global i32 101 #0
@llvm.fake_var = internal global i32 123
@qux = external global i64

define void @cold_func(i32 %0) !prof !15 {
  %2 = load i32, ptr @cold_bss
  %3 = load i32, ptr @cold_data
  %4 = srem i32 %0, 2
  %5 = sext i32 %4 to i64
  %6 = getelementptr inbounds [2 x ptr], ptr @cold_relro_array, i64 0, i64 %5
  %7 = load ptr, ptr %6
  %8 = load i32, ptr %7
  %9 = load i32, ptr @data_with_unknown_hotness
  %11 = load i32, ptr @hot_data
  %12 = load i32, ptr @cold_data_custom_foo_section
  %13 = call i32 (...) @func_taking_arbitrary_param(ptr @.str.2, i32 %2, i32 %3, i32 %8, i32 %9, i32 %11, i32 %12)
  ret void
}

define i32 @unprofiled_func() {
  %a = load i32, ptr @data_with_unknown_hotness
  %b = load i32, ptr @hot_data
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
  %8 = load i32, ptr @hot_data
  %9 = load i32, ptr @hot_bss
  %10 = load i32, ptr @hot_data_custom_bar_section
  %11 = call i32 (...) @func_taking_arbitrary_param(ptr @.str.1, i32 %7, i32 %8, i32 %9, i32 %10)
  ret void
}

define i32 @main(i32 %0, ptr %1) !prof !15 {
  br label %11

5:                                                ; preds = %11
  %6 = call i32 @rand()
  store i32 %6, ptr @cold_bss
  store i32 %6, ptr @cold_data
  store i32 %6, ptr @bss2
  store i32 %6, ptr @data3
  call void @cold_func(i32 %6)
  ret i32 0

11:                                               ; preds = %11, %2
  %12 = phi i32 [ 0, %2 ], [ %19, %11 ]
  %13 = call i32 @rand()
  %14 = srem i32 %13, 2
  %15 = sext i32 %14 to i64
  %16 = getelementptr inbounds [2 x ptr], ptr @hot_relro_array, i64 0, i64 %15
  %17 = load ptr, ptr %16
  store i32 %13, ptr %17
  store i32 %13, ptr @hot_data
  %18 = add i32 %13, 1
  store i32 %18, ptr @hot_bss
  call void @hot_func(i32 %12)
  %19 = add i32 %12, 1
  %20 = icmp eq i32 %19, 100000
  br i1 %20, label %5, label %11, !prof !16
}

declare i32 @rand()
declare i32 @func_taking_arbitrary_param(...)

attributes #0 = {"data-section"="bar"}

!llvm.module.flags = !{!1}

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
