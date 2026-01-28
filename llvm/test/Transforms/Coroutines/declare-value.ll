;RUN: opt -mtriple='arm64-' %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s

; CHECK-LABEL: define swifttailcc void @coroutineA
; CHECK-SAME:    (ptr swiftasync %[[frame_ptr:.*]],
; CHECK:  %.debug = alloca double, align 8
; CHECK-NEXT:    #dbg_declare(ptr %{{.*}}, !{{[0-9]+}}, !DIExpression(DW_OP_deref), !{{[0-9]+}})
; CHECK-NEXT:  store double %{{[0-9]+}}, ptr %{{.*}}, align 8
; CHECK:       %[[frame_ptr_alloca:.*]] = alloca ptr,
; CHECK-NEXT:  #dbg_declare(ptr %[[frame_ptr_alloca]], !{{[0-9]+}}, !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 24), !{{[0-9]+}})
; CHECK-NEXT:  store ptr %[[frame_ptr]], ptr %[[frame_ptr_alloca]]

; ModuleID = '/Users/srastogi/Development/llvm-project-2/llvm/test/Transforms/Coroutines/declare-value.ll'
source_filename = "/Users/srastogi/Development/llvm-project-2/llvm/test/Transforms/Coroutines/declare-value.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-unknown"

@coroutineATu = global <{ i32, i32 }> <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @coroutineA to i64), i64 ptrtoint (ptr @coroutineATu to i64)) to i32), i32 16 }>, align 8

; Function Attrs: presplitcoroutine
define swifttailcc void @coroutineA(ptr swiftasync %arg, double %0) #0 !dbg !1 {
  %var_with_dbg_value = alloca ptr, align 8
  %var_with_dbg_declare = alloca ptr, align 8
    #dbg_declare(ptr %var_with_dbg_declare, !5, !DIExpression(), !7)
    #dbg_declare_value(double %0, !5, !DIExpression(), !7)
  %i2 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr nonnull @coroutineATu)
  %i3 = call ptr @llvm.coro.begin(token %i2, ptr null)
  %i7 = call ptr @llvm.coro.async.resume(), !dbg !7
  %i10 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %i7, ptr nonnull @__swift_async_resume_get_context, ptr nonnull @coroutineA.1, ptr %i7, i64 0, i64 0, ptr %arg), !dbg !7
  call void @dont_optimize(ptr %var_with_dbg_value, ptr %var_with_dbg_declare), !dbg !7
  unreachable, !dbg !7
}

define weak_odr hidden ptr @__swift_async_resume_get_context(ptr %arg) !dbg !8 {
  ret ptr %arg, !dbg !9
}

define hidden swifttailcc void @coroutineA.1(ptr %arg, i64 %arg1, i64 %arg2, ptr %arg3) !dbg !10 {
  ret void, !dbg !11
}

declare void @dont_optimize(ptr, ptr)

; Function Attrs: nomerge nounwind
declare ptr @llvm.coro.async.resume() #1

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #2

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, ptr) #2

; Function Attrs: nomerge nounwind
declare { ptr } @llvm.coro.suspend.async.sl_p0s(i32, ptr, ptr, ...) #1

attributes #0 = { presplitcoroutine }
attributes #1 = { nomerge nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram(scope: null, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!2 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !DIFile(filename: "blah", directory: "")
!4 = !{}
!5 = !DILocalVariable(scope: !1, type: !6)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Klass")
!7 = !DILocation(line: 0, scope: !1)
!8 = distinct !DISubprogram(scope: null, spFlags: DISPFlagDefinition, unit: !2)
!9 = !DILocation(line: 0, scope: !8)
!10 = distinct !DISubprogram(scope: null, spFlags: DISPFlagDefinition, unit: !2)
!11 = !DILocation(line: 0, scope: !10)