; RUN: opt --passes=simplifycfg < %s | FileCheck %s

; CHECK: anything

; ModuleID = 'repro.2beeda155604e689-cgu.0'
source_filename = "repro.2beeda155604e689-cgu.0"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@alloc_c9aa3264c948ed374957d673a2181758 = private unnamed_addr constant [8 x i8] c"repro.rs", align 1
@alloc_6ecb7eed331e379713836616fde14f5d = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_c9aa3264c948ed374957d673a2181758, [16 x i8] c"\08\00\00\00\00\00\00\00\12\00\00\00\11\00\00\00" }>, align 8
@alloc_7fab8898959a1cc23912ee68ba0a857a = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_c9aa3264c948ed374957d673a2181758, [16 x i8] c"\08\00\00\00\00\00\00\00\12\00\00\00\05\00\00\00" }>, align 8
@alloc_10329d630a8a553b05d2a51b598996f5 = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_c9aa3264c948ed374957d673a2181758, [16 x i8] c"\08\00\00\00\00\00\00\00\13\00\00\00\05\00\00\00" }>, align 8

; repro::repro_func
; Function Attrs: sspstrong uwtable
define void @_RNvCs3LQYlzOoal9_5repro10repro_func(ptr align 1 %shapes.0, i64 %shapes.1) unnamed_addr #0 {
start:
; call repro::repro_generic::<repro::ReproShape>
  call void @_RINvCs3LQYlzOoal9_5repro13repro_genericNtB2_10ReproShapeEB2_(ptr align 1 %shapes.0, i64 %shapes.1)
  ret void
}

; repro::repro_generic::<repro::ReproShape>
; Function Attrs: sspstrong uwtable
define internal void @_RINvCs3LQYlzOoal9_5repro13repro_genericNtB2_10ReproShapeEB2_(ptr align 1 %shapes.0, i64 %shapes.1) unnamed_addr #0 personality ptr @__CxxFrameHandler3 {
start:
  %_18 = alloca [1 x i8], align 1
  %_17 = alloca [1 x i8], align 1
  store i8 0, ptr %_18, align 1
  store i8 0, ptr %_17, align 1
  %_7 = icmp ult i64 1, %shapes.1
  br i1 %_7, label %bb1, label %panic

bb1:                                              ; preds = %start
  %_3 = getelementptr inbounds nuw i8, ptr %shapes.0, i64 1
  store i8 1, ptr %_18, align 1
; call <repro::ShapeInfo<repro::ReproShape> as core::clone::Clone>::clone
  %_2 = call zeroext i1 @_RNvXCs3LQYlzOoal9_5reproINtB2_9ShapeInfoNtB2_10ReproShapeENtNtCshEuXqvZjEXJ_4core5clone5Clone5cloneB2_(ptr align 1 %_3)
  %_11 = icmp ult i64 0, %shapes.1
  br i1 %_11, label %bb3, label %panic1

panic:                                            ; preds = %start
; call core::panicking::panic_bounds_check
  call void @_RNvNtCshEuXqvZjEXJ_4core9panicking18panic_bounds_check(i64 1, i64 %shapes.1, ptr align 8 @alloc_6ecb7eed331e379713836616fde14f5d) #4
  unreachable

bb3:                                              ; preds = %bb1
  br label %bb4

panic1:                                           ; preds = %bb1
; invoke core::panicking::panic_bounds_check
  invoke void @_RNvNtCshEuXqvZjEXJ_4core9panicking18panic_bounds_check(i64 0, i64 %shapes.1, ptr align 8 @alloc_7fab8898959a1cc23912ee68ba0a857a) #4
          to label %unreachable unwind label %funclet_bb14

bb14:                                             ; preds = %funclet_bb14
  %0 = load i8, ptr %_18, align 1
  %1 = trunc nuw i8 %0 to i1
  br i1 %1, label %bb13, label %bb14_cleanup_trampoline_bb10

funclet_bb14:                                     ; preds = %bb5, %panic1
  %cleanuppad = cleanuppad within none []
  br label %bb14

unreachable:                                      ; preds = %panic2, %panic1
  unreachable

bb4:                                              ; preds = %bb3
  store i8 0, ptr %_18, align 1
  %2 = getelementptr inbounds nuw i8, ptr %shapes.0, i64 0
  %3 = zext i1 %_2 to i8
  store i8 %3, ptr %2, align 1
  store i8 0, ptr %_18, align 1
  store i8 1, ptr %_17, align 1
; call <repro::ReproShape as core::default::Default>::default
  %_12 = call zeroext i1 @_RNvXs_Cs3LQYlzOoal9_5reproNtB4_10ReproShapeNtNtCshEuXqvZjEXJ_4core7default7Default7defaultB4_()
  %_16 = icmp ult i64 1, %shapes.1
  br i1 %_16, label %bb7, label %panic2

bb7:                                              ; preds = %bb4
  br label %bb8

panic2:                                           ; preds = %bb4
; invoke core::panicking::panic_bounds_check
  invoke void @_RNvNtCshEuXqvZjEXJ_4core9panicking18panic_bounds_check(i64 1, i64 %shapes.1, ptr align 8 @alloc_10329d630a8a553b05d2a51b598996f5) #4
          to label %unreachable unwind label %funclet_bb12

bb12:                                             ; preds = %funclet_bb12
  %4 = load i8, ptr %_17, align 1
  %5 = trunc nuw i8 %4 to i1
  br i1 %5, label %bb11, label %bb12_cleanup_trampoline_bb10

funclet_bb12:                                     ; preds = %bb9, %panic2
  %cleanuppad3 = cleanuppad within none []
  br label %bb12

bb8:                                              ; preds = %bb7
  store i8 0, ptr %_17, align 1
  %6 = getelementptr inbounds nuw i8, ptr %shapes.0, i64 1
  %7 = zext i1 %_12 to i8
  store i8 %7, ptr %6, align 1
  store i8 0, ptr %_17, align 1
  ret void

bb9:                                              ; preds = %funclet_bb9
  store i8 0, ptr %_17, align 1
  %8 = getelementptr inbounds nuw i8, ptr %shapes.0, i64 1
  %9 = zext i1 %_12 to i8
  store i8 %9, ptr %8, align 1
  cleanupret from %cleanuppad4 unwind label %funclet_bb12

funclet_bb9:                                      ; No predecessors!
  %cleanuppad4 = cleanuppad within none []
  br label %bb9

bb10:                                             ; preds = %funclet_bb10
  cleanupret from %cleanuppad5 unwind to caller

funclet_bb10:                                     ; preds = %bb13, %bb14_cleanup_trampoline_bb10, %bb11, %bb12_cleanup_trampoline_bb10
  %cleanuppad5 = cleanuppad within none []
  br label %bb10

bb12_cleanup_trampoline_bb10:                     ; preds = %bb12
  cleanupret from %cleanuppad3 unwind label %funclet_bb10

bb11:                                             ; preds = %bb12
  cleanupret from %cleanuppad3 unwind label %funclet_bb10

bb5:                                              ; preds = %funclet_bb5
  store i8 0, ptr %_18, align 1
  %10 = getelementptr inbounds nuw i8, ptr %shapes.0, i64 0
  %11 = zext i1 %_2 to i8
  store i8 %11, ptr %10, align 1
  cleanupret from %cleanuppad6 unwind label %funclet_bb14

funclet_bb5:                                      ; No predecessors!
  %cleanuppad6 = cleanuppad within none []
  br label %bb5

bb14_cleanup_trampoline_bb10:                     ; preds = %bb14
  cleanupret from %cleanuppad unwind label %funclet_bb10

bb13:                                             ; preds = %bb14
  cleanupret from %cleanuppad unwind label %funclet_bb10
}

; <repro::ShapeInfo<repro::ReproShape> as core::clone::Clone>::clone
; Function Attrs: inlinehint sspstrong uwtable
define zeroext i1 @_RNvXCs3LQYlzOoal9_5reproINtB2_9ShapeInfoNtB2_10ReproShapeENtNtCshEuXqvZjEXJ_4core5clone5Clone5cloneB2_(ptr align 1 %self) unnamed_addr #1 {
start:
; call <repro::ReproShape as core::clone::Clone>::clone
  %_2 = call zeroext i1 @_RNvXs1_Cs3LQYlzOoal9_5reproNtB5_10ReproShapeNtNtCshEuXqvZjEXJ_4core5clone5Clone5cloneB5_(ptr align 1 %self)
  ret i1 %_2
}

; <repro::ReproShape as core::default::Default>::default
; Function Attrs: inlinehint sspstrong uwtable
define internal zeroext i1 @_RNvXs_Cs3LQYlzOoal9_5reproNtB4_10ReproShapeNtNtCshEuXqvZjEXJ_4core7default7Default7defaultB4_() unnamed_addr #1 {
start:
  %_0 = alloca [1 x i8], align 1
  store i8 0, ptr %_0, align 1
  %0 = load i8, ptr %_0, align 1
  %1 = trunc nuw i8 %0 to i1
  ret i1 %1
}

; <repro::ReproShape as core::clone::Clone>::clone
; Function Attrs: inlinehint sspstrong uwtable
define internal zeroext i1 @_RNvXs1_Cs3LQYlzOoal9_5reproNtB5_10ReproShapeNtNtCshEuXqvZjEXJ_4core5clone5Clone5cloneB5_(ptr align 1 %self) unnamed_addr #1 {
start:
  %0 = load i8, ptr %self, align 1
  %_0 = trunc nuw i8 %0 to i1
  ret i1 %_0
}

declare i32 @__CxxFrameHandler3(...) unnamed_addr #2

; core::panicking::panic_bounds_check
; Function Attrs: cold minsize noinline noreturn optsize sspstrong uwtable
declare void @_RNvNtCshEuXqvZjEXJ_4core9panicking18panic_bounds_check(i64, i64, ptr align 8) unnamed_addr #3

attributes #0 = { sspstrong uwtable "target-cpu"="x86-64" "target-features"="+cx16,+sse3,+sahf" }
attributes #1 = { inlinehint sspstrong uwtable "target-cpu"="x86-64" "target-features"="+cx16,+sse3,+sahf" }
attributes #2 = { "target-cpu"="x86-64" }
attributes #3 = { cold minsize noinline noreturn optsize sspstrong uwtable "target-cpu"="x86-64" "target-features"="+cx16,+sse3,+sahf" }
attributes #4 = { noreturn }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 2, !"cfguard", i32 2}
!2 = !{!"rustc version 1.88.0 (d4e9f376db 2025-06-24) (1.88.0-ms-20250625.1+d4e9f376db)"}
