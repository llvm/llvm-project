

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions \
; RUN:     -partition-static-data-sections=true -data-sections=true \
; RUN:     -unique-section-names=true -relocation-model=pic \
; RUN:     %s -o - 2>&1 | FileCheck %s --check-prefixes=SYM,DATA

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions \
; RUN:     -partition-static-data-sections=true -data-sections=true \
; RUN:     -unique-section-names=false -relocation-model=pic \
; RUN:     %s -o - 2>&1 | FileCheck %s --check-prefixes=UNIQ,DATA

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions \
; RUN:     -partition-static-data-sections=true -data-sections=false \
; RUN:     -unique-section-names=false -relocation-model=pic \
; RUN:     %s -o - 2>&1 | FileCheck %s --check-prefixes=AGG,DATA

; SYM: .section .rodata.str1.1.hot.
; UNIQ: .section	.rodata.str1.1.hot.,"aMS",@progbits,1
; AGG: .section	.rodata.str1.1.hot
; DATA: .L.str
; DATA:    "hot\t"
; DATA: .L.str.1
; DATA:    "%d\t%d\t%d\n"


; SYM:  .section	.data.rel.ro.hot.hot_relro_array
; SYM: .section	.data.hot.hot_data,"aw",@progbits
; SYM: .section	.bss.hot.hot_bss,"aw",@nobits

; UNIQ: .section	.data.rel.ro.hot.,"aw",@progbits,unique,3
; UNIQ: .section	.data.hot.,"aw",@progbits,unique,4
; UNIQ: .section	.bss.hot.,"aw",@nobits,unique,5

; AGG: .section	.data.rel.ro.hot.,"aw",@progbits
; AGG: .section	.data.hot.,"aw",@progbits
; AGG: .section .bss.hot.,"aw",@nobits


; SYM: .section	.rodata.str1.1.unlikely.,"aMS",@progbits,1
; UNIQ: section	.rodata.str1.1.unlikely.,"aMS",@progbits,1
; AGG: .section	.rodata.str1.1.unlikely.,"aMS",@progbits,1
; DATA: .L.str.2:
; DATA:    "cold%d\t%d\t%d\n"


; SYM: .section	.bss.unlikely.cold_bss,"aw",@nobits
; SYM: .section	.data.unlikely.cold_data,"aw",@progbits
; SYM: .section	.data.rel.ro.unlikely.cold_relro_array,"aw",@progbits
; SYM: .section	.bss.unlikely._ZL4bss2,"aw",@nobits
; SYM: .section	.data.unlikely._ZL5data3,"aw",@progbits

; UNIQ: .section	.bss.unlikely.,"aw",@nobits,unique,6
; UNIQ: .section	.data.unlikely.,"aw",@progbits,unique,7
; UNIQ: .section	.data.rel.ro.unlikely.,"aw",@progbits,unique,8
; UNIQ: .section	.bss.unlikely.,"aw",@nobits,unique,9
; UNIQ: .section	.data.unlikely.,"aw",@progbits,unique,10

; AGG: .section	.bss.unlikely.,"aw",@nobits
; AGG: .section	.data.unlikely.,"aw",@progbits
; AGG: .section	.data.rel.ro.unlikely.,"aw",@progbits
; AGG: .section	.bss.unlikely.,"aw",@nobits
; AGG: .section	.data.unlikely.,"aw",@progbits

@.str = private unnamed_addr constant [5 x i8] c"hot\09\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"%d\09%d\09%d\0A\00", align 1
@hot_relro_array = internal constant [2 x ptr] [ptr @_ZL4bss2, ptr @_ZL5data3]
@hot_data = internal global i32 5
@hot_bss = internal global i32 0
@.str.2 = private unnamed_addr constant [14 x i8] c"cold%d\09%d\09%d\0A\00", align 1
@cold_bss = internal global i32 0
@cold_data = internal global i32 4
@cold_relro_array = internal constant [2 x ptr] [ptr @_ZL5data3, ptr @_ZL4bss2]
@_ZL4bss2 = internal global i32 0
@_ZL5data3 = internal global i32 3

define void @hot_callee(i32 %0) !prof !51 {
  %2 = call i32 (ptr, ...) @printf(ptr @.str)
  %3 = srem i32 %0, 2
  %4 = sext i32 %3 to i64
  %5 = getelementptr inbounds [2 x ptr], ptr @hot_relro_array, i64 0, i64 %4
  %6 = load ptr, ptr %5
  %7 = load i32, ptr %6
  %8 = load i32, ptr @hot_data
  %9 = load i32, ptr @hot_bss
  %10 = call i32 (ptr, ...) @printf(ptr @.str.1, i32 %7, i32 %8, i32 %9)
  ret void
}

define void @cold_callee(i32 %0) !prof !52 {
  %2 = load i32, ptr @cold_bss
  %3 = load i32, ptr @cold_data
  %4 = srem i32 %0, 2
  %5 = sext i32 %4 to i64
  %6 = getelementptr inbounds [2 x ptr], ptr @cold_relro_array, i64 0, i64 %5
  %7 = load ptr, ptr %6
  %8 = load i32, ptr %7
  %9 = call i32 (ptr, ...) @printf(ptr @.str.2, i32 %2, i32 %3, i32 %8)
  ret void
}

define i32 @main(i32 %0, ptr %1) !prof !52 {
  %3 = call i64 @time(ptr null)
  %4 = trunc i64 %3 to i32
  call void @srand(i32 %4)
  br label %11

5:                                                ; preds = %11
  %6 = call i32 @rand()
  store i32 %6, ptr @cold_bss
  store i32 %6, ptr @cold_data
  store i32 %6, ptr @_ZL4bss2
  store i32 %6, ptr @_ZL5data3
  call void @cold_callee(i32 %6)
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
  call void @hot_callee(i32 %12)
  %19 = add i32 %12, 1
  %20 = icmp eq i32 %19, 100000
  br i1 %20, label %5, label %11, !prof !53
}

declare void @srand(i32)
declare i64 @time(ptr)
declare i32 @rand()
declare i32 @printf(ptr, ...)

!llvm.module.flags = !{!12}

!12 = !{i32 1, !"ProfileSummary", !13}
!13 = !{!14, !15, !16, !17, !18, !19, !20, !23}
!14 = !{!"ProfileFormat", !"InstrProf"}
!15 = !{!"TotalCount", i64 1460183}
!16 = !{!"MaxCount", i64 849024}
!17 = !{!"MaxInternalCount", i64 32769}
!18 = !{!"MaxFunctionCount", i64 849024}
!19 = !{!"NumCounts", i64 23627}
!20 = !{!"NumFunctions", i64 3271}
!23 = !{!"DetailedSummary", !24}
!24 = !{!36, !40}
!36 = !{i32 990000, i64 166, i32 73}
!40 = !{i32 999999, i64 1, i32 1443}
!51 = !{!"function_entry_count", i64 100000}
!52 = !{!"function_entry_count", i64 1}
!53 = !{!"branch_weights", i32 1, i32 99999}
