; Tests whether we properly setup llvm.dbg.addr for Swift.
;
; Since we do not have any guarantees around the usage of llvm.dbg.addr, we can
; not propagate them like we do llvm.dbg.declare into funclets. But if users
; create the debug_value for us, make sure that we propagate llvm.dbg.addr into
; the beginning coroutine and all other funclets.

; RUN: opt %s -passes='function(coro-early),cgscc(coro-split,simplifycfg)' -S | FileCheck %s

; CHECK-LABEL: define swifttailcc void @"$s10async_args14withGenericArgyyxnYalF"(%swift.context* swiftasync %0, %swift.opaque* noalias %1, %swift.type* %T){{.*}} {
; CHECK: call void @llvm.dbg.declare(
; CHECK: llvm.dbg.addr
; CHECK-NOT: llvm.dbg.value
; CHECK-NOT: llvm.dbg.addr
; CHECK-NOT: llvm.dbg.declare
; CHECK: musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %19, i8* bitcast (void (i8*)* @"$s10async_args14withGenericArgyyxnYalFTY0_" to i8*), i64 0, i64 0)
; CHECK-NEXT: ret void
; CHECK-NEXT: }

; CHECK-LABEL: define internal swifttailcc void @"$s10async_args14withGenericArgyyxnYalFTY0_"(i8* swiftasync %0)
; CHECK: entryresume.0
; CHECK-NEXT: %.debug
; CHECK-NEXT: call void @llvm.dbg.declare(
; CHECK: llvm.dbg.addr
; CHECK: musttail call swifttailcc void @"$s10async_args10forceSplityyYaF"(%swift.context* swiftasync
; CHECK-NEXT: ret void
; CHECK-NEXT: }

; CHECK: define internal swifttailcc void @"$s10async_args14withGenericArgyyxnYalFTQ1_"(i8* swiftasync %0)
; CHECK: llvm.dbg.declare
; CHECK: llvm.dbg.addr
; CHECK: llvm.dbg.value(metadata %swift.opaque** undef,
; CHECK: ret void
; CHECK-NEXT: }

; ModuleID = 'async_args.ll'
source_filename = "async_args.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.opaque = type opaque
%swift.type = type { i64 }
%swift.context = type { %swift.context*, void (%swift.context*)*, i64 }
%swift.vwtable = type { i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, i64, i32, i32 }

@"$s10async_args10forceSplityyYaFTu" = global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.context*)* @"$s10async_args10forceSplityyYaF" to i64), i64 ptrtoint (%swift.async_func_pointer* @"$s10async_args10forceSplityyYaFTu" to i64)) to i32), i32 20 }>, align 8
@"$s10async_args14withGenericArgyyxnYalFTu" = global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.context*, %swift.opaque*, %swift.type*)* @"$s10async_args14withGenericArgyyxnYalF" to i64), i64 ptrtoint (%swift.async_func_pointer* @"$s10async_args14withGenericArgyyxnYalFTu" to i64)) to i32), i32 20 }>, align 8
@"_swift_FORCE_LOAD_$_swiftCompatibilityConcurrency_$_async_args" = weak_odr hidden constant void ()* @"_swift_FORCE_LOAD_$_swiftCompatibilityConcurrency"
@__swift_reflection_version = linkonce_odr hidden constant i16 3
@swift_async_extendedFramePointerFlags = extern_weak global i8*
@_swift_async_extendedFramePointerFlagsUser = linkonce_odr hidden global i8** @swift_async_extendedFramePointerFlags
@llvm.used = appending global [10 x i8*] [i8* bitcast (void (%swift.opaque*, %swift.type*)* @"$s10async_args3useyyxlF" to i8*), i8* bitcast (void (%swift.opaque*, %swift.type*)* @"$s10async_args4use2yyxlF" to i8*), i8* bitcast (void (%swift.context*)* @"$s10async_args10forceSplityyYaF" to i8*), i8* bitcast (%swift.async_func_pointer* @"$s10async_args10forceSplityyYaFTu" to i8*), i8* bitcast (void (%swift.opaque*, %swift.type*)* @"$s10async_args4use3yyxlF" to i8*), i8* bitcast (void (%swift.context*, %swift.opaque*, %swift.type*)* @"$s10async_args14withGenericArgyyxnYalF" to i8*), i8* bitcast (%swift.async_func_pointer* @"$s10async_args14withGenericArgyyxnYalFTu" to i8*), i8* bitcast (void ()** @"_swift_FORCE_LOAD_$_swiftCompatibilityConcurrency_$_async_args" to i8*), i8* bitcast (i16* @__swift_reflection_version to i8*), i8* bitcast (i8*** @_swift_async_extendedFramePointerFlagsUser to i8*)], section "llvm.metadata"

define hidden swiftcc i1 @"$s10async_args7booleanSbvg"() #0 !dbg !31 {
entry:
  ret i1 false, !dbg !37
}

define swiftcc void @"$s10async_args3useyyxlF"(%swift.opaque* noalias nocapture %0, %swift.type* %T) #0 !dbg !39 {
entry:
  %T1 = alloca %swift.type*, align 8
  %t.debug = alloca %swift.opaque*, align 8
  %1 = bitcast %swift.opaque** %t.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  store %swift.type* %T, %swift.type** %T1, align 8
  call void @llvm.dbg.declare(metadata %swift.type** %T1, metadata !45, metadata !DIExpression()), !dbg !52
  store %swift.opaque* %0, %swift.opaque** %t.debug, align 8, !dbg !52
  call void @llvm.dbg.addr(metadata %swift.opaque** %t.debug, metadata !50, metadata !DIExpression(DW_OP_deref)), !dbg !53
  ret void, !dbg !54
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #2

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.addr(metadata, metadata, metadata) #1

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

define swiftcc void @"$s10async_args4use2yyxlF"(%swift.opaque* noalias nocapture %0, %swift.type* %T) #0 !dbg !56 {
entry:
  %T1 = alloca %swift.type*, align 8
  %t.debug = alloca %swift.opaque*, align 8
  %1 = bitcast %swift.opaque** %t.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  store %swift.type* %T, %swift.type** %T1, align 8
  call void @llvm.dbg.declare(metadata %swift.type** %T1, metadata !58, metadata !DIExpression()), !dbg !60
  store %swift.opaque* %0, %swift.opaque** %t.debug, align 8, !dbg !60
  call void @llvm.dbg.addr(metadata %swift.opaque** %t.debug, metadata !59, metadata !DIExpression(DW_OP_deref)), !dbg !61
  ret void, !dbg !62
}

declare swifttailcc void @"$s10async_args10forceSplityyYaF"(%swift.context* swiftasync %0) #0

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, i8*) #3

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #4

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #3

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_1(i8* %0, %swift.context* %1) #3 !dbg !69 {
entry:
  %2 = bitcast i8* %0 to void (%swift.context*)*, !dbg !71
  musttail call swifttailcc void %2(%swift.context* swiftasync %1), !dbg !71
  ret void, !dbg !71
}

; Function Attrs: nounwind
declare i1 @llvm.coro.end.async(i8*, i1, ...) #3

define swiftcc void @"$s10async_args4use3yyxlF"(%swift.opaque* noalias nocapture %0, %swift.type* %T) #0 !dbg !72 {
entry:
  %T1 = alloca %swift.type*, align 8
  %t.debug = alloca %swift.opaque*, align 8
  %1 = bitcast %swift.opaque** %t.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %1, i8 0, i64 8, i1 false)
  store %swift.type* %T, %swift.type** %T1, align 8
  call void @llvm.dbg.declare(metadata %swift.type** %T1, metadata !74, metadata !DIExpression()), !dbg !76
  store %swift.opaque* %0, %swift.opaque** %t.debug, align 8, !dbg !76
  call void @llvm.dbg.addr(metadata %swift.opaque** %t.debug, metadata !75, metadata !DIExpression(DW_OP_deref)), !dbg !77
  ret void, !dbg !78
}

define swifttailcc void @"$s10async_args14withGenericArgyyxnYalF"(%swift.context* swiftasync %0, %swift.opaque* noalias nocapture %1, %swift.type* %T) #0 !dbg !80 {
entry:
  call void @llvm.dbg.declare(metadata %swift.type* %T, metadata !82, metadata !DIExpression()), !dbg !84
  %2 = alloca %swift.context*, align 8
  %msg.debug = alloca %swift.opaque*, align 8
  %3 = bitcast %swift.context* %0 to <{ %swift.context*, void (%swift.context*)*, i32 }>*
  %4 = call token @llvm.coro.id.async(i32 20, i32 16, i32 0, i8* bitcast (%swift.async_func_pointer* @"$s10async_args14withGenericArgyyxnYalFTu" to i8*))
  %5 = call i8* @llvm.coro.begin(token %4, i8* null)
  store %swift.context* %0, %swift.context** %2, align 8
  %6 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %6, i8 0, i64 8, i1 false)
  %7 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %7, i8 0, i64 8, i1 false)
  %8 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %8, i8 0, i64 8, i1 false)
  %9 = bitcast %swift.type* %T to i8***, !dbg !85
  %10 = getelementptr inbounds i8**, i8*** %9, i64 -1, !dbg !85
  %T.valueWitnesses = load i8**, i8*** %10, align 8, !dbg !85, !invariant.load !36, !dereferenceable !88
  %11 = bitcast i8** %T.valueWitnesses to %swift.vwtable*, !dbg !85
  %12 = getelementptr inbounds %swift.vwtable, %swift.vwtable* %11, i32 0, i32 8, !dbg !85
  %size = load i64, i64* %12, align 8, !dbg !85, !invariant.load !36
  %13 = add i64 %size, 15, !dbg !85
  %14 = and i64 %13, -16, !dbg !85
  %15 = call swiftcc i8* @swift_task_alloc(i64 %14) #3, !dbg !85
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %15), !dbg !85
  %16 = bitcast i8* %15 to %swift.opaque*, !dbg !85
  store %swift.opaque* %1, %swift.opaque** %msg.debug, align 8, !dbg !84
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !89
  call void @llvm.dbg.addr(metadata %swift.opaque** %msg.debug, metadata !83, metadata !DIExpression(DW_OP_deref)), !dbg !91
  %17 = call i8* @llvm.coro.async.resume(), !dbg !84
  %18 = load %swift.context*, %swift.context** %2, align 8, !dbg !84
  %19 = load %swift.context*, %swift.context** %2, align 8, !dbg !84
  %20 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %17, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, i64, i64, %swift.context*)* @__swift_suspend_point to i8*), i8* %17, i64 0, i64 0, %swift.context* %19), !dbg !84
  %21 = extractvalue { i8* } %20, 0, !dbg !84
  %22 = call i8* @__swift_async_resume_get_context(i8* %21), !dbg !84
  %23 = bitcast i8* %22 to %swift.context*, !dbg !84
  store %swift.context* %23, %swift.context** %2, align 8, !dbg !84
  store %swift.opaque* %1, %swift.opaque** %msg.debug, align 8, !dbg !84
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !89
  call void @llvm.dbg.addr(metadata %swift.opaque** %msg.debug, metadata !83, metadata !DIExpression(DW_OP_deref)), !dbg !91
  %24 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 2, !dbg !92
  %25 = load i8*, i8** %24, align 8, !dbg !92, !invariant.load !36
  %initializeWithCopy = bitcast i8* %25 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !92
  %26 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %16, %swift.opaque* noalias %1, %swift.type* %T) #3, !dbg !92
  call swiftcc void @"$s10async_args4use3yyxlF"(%swift.opaque* noalias nocapture %16, %swift.type* %T), !dbg !93
  %27 = getelementptr inbounds i8*, i8** %T.valueWitnesses, i32 1, !dbg !93
  %28 = load i8*, i8** %27, align 8, !dbg !93, !invariant.load !36
  %destroy = bitcast i8* %28 to void (%swift.opaque*, %swift.type*)*, !dbg !93
  call void %destroy(%swift.opaque* noalias %16, %swift.type* %T) #3, !dbg !93
  %29 = load i32, i32* getelementptr inbounds (%swift.async_func_pointer, %swift.async_func_pointer* @"$s10async_args10forceSplityyYaFTu", i32 0, i32 1), align 8, !dbg !94
  %30 = zext i32 %29 to i64, !dbg !94
  %31 = call swiftcc i8* @swift_task_alloc(i64 %30) #3, !dbg !94
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %31), !dbg !94
  %32 = bitcast i8* %31 to <{ %swift.context*, void (%swift.context*)*, i32 }>*, !dbg !94
  %33 = load %swift.context*, %swift.context** %2, align 8, !dbg !94
  %34 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %32, i32 0, i32 0, !dbg !94
  store %swift.context* %33, %swift.context** %34, align 8, !dbg !94
  %35 = call i8* @llvm.coro.async.resume(), !dbg !94
  %36 = bitcast i8* %35 to void (%swift.context*)*, !dbg !94
  %37 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %32, i32 0, i32 1, !dbg !94
  store void (%swift.context*)* %36, void (%swift.context*)** %37, align 8, !dbg !94
  %38 = bitcast i8* %31 to %swift.context*, !dbg !94
  %39 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %35, i8* bitcast (i8* (i8*)* @__swift_async_resume_project_context to i8*), i8* bitcast (void (i8*, %swift.context*)* @__swift_suspend_dispatch_1.1 to i8*), i8* bitcast (void (%swift.context*)* @"$s10async_args10forceSplityyYaF" to i8*), %swift.context* %38), !dbg !94
  %40 = extractvalue { i8* } %39, 0, !dbg !94
  %41 = call i8* @__swift_async_resume_project_context(i8* %40), !dbg !94
  %42 = bitcast i8* %41 to %swift.context*, !dbg !94
  store %swift.context* %42, %swift.context** %2, align 8, !dbg !94
  call swiftcc void @swift_task_dealloc(i8* %31) #3, !dbg !94
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %31), !dbg !94
  %43 = call i8* @llvm.coro.async.resume(), !dbg !94
  %44 = load %swift.context*, %swift.context** %2, align 8, !dbg !94
  %45 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %43, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, i64, i64, %swift.context*)* @__swift_suspend_point to i8*), i8* %43, i64 0, i64 0, %swift.context* %44), !dbg !94
  %46 = extractvalue { i8* } %45, 0, !dbg !94
  %47 = call i8* @__swift_async_resume_get_context(i8* %46), !dbg !94
  %48 = bitcast i8* %47 to %swift.context*, !dbg !94
  store %swift.context* %48, %swift.context** %2, align 8, !dbg !94
  store %swift.opaque* %1, %swift.opaque** %msg.debug, align 8, !dbg !84
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !89
  call void @llvm.dbg.addr(metadata %swift.opaque** %msg.debug, metadata !83, metadata !DIExpression(DW_OP_deref)), !dbg !91
  %49 = call swiftcc i1 @"$s10async_args7booleanSbvg"(), !dbg !95
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !95
  br i1 %49, label %50, label %52, !dbg !95

50:                                               ; preds = %entry
  %51 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %16, %swift.opaque* noalias %1, %swift.type* %T) #3, !dbg !97
  call swiftcc void @"$s10async_args3useyyxlF"(%swift.opaque* noalias nocapture %16, %swift.type* %T), !dbg !99
  call void %destroy(%swift.opaque* noalias %16, %swift.type* %T) #3, !dbg !100
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !100
  call void @llvm.dbg.value(metadata %swift.opaque** undef, metadata !83, metadata !DIExpression()), !dbg !91
  br label %54, !dbg !100

52:                                               ; preds = %entry
  %53 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %16, %swift.opaque* noalias %1, %swift.type* %T) #3, !dbg !101
  call swiftcc void @"$s10async_args4use2yyxlF"(%swift.opaque* noalias nocapture %16, %swift.type* %T), !dbg !103
  call void %destroy(%swift.opaque* noalias %16, %swift.type* %T) #3, !dbg !104
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !104
  br label %54, !dbg !104

54:                                               ; preds = %50, %52
  call void %destroy(%swift.opaque* noalias %1, %swift.type* %T) #3, !dbg !105
  %55 = bitcast %swift.opaque* %16 to i8*, !dbg !105
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %55), !dbg !105
  call swiftcc void @swift_task_dealloc(i8* %15) #3, !dbg !105
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !105
  %56 = load %swift.context*, %swift.context** %2, align 8, !dbg !105
  %57 = bitcast %swift.context* %56 to <{ %swift.context*, void (%swift.context*)*, i32 }>*, !dbg !105
  %58 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %57, i32 0, i32 1, !dbg !105
  %59 = load void (%swift.context*)*, void (%swift.context*)** %58, align 8, !dbg !105
  %60 = load %swift.context*, %swift.context** %2, align 8, !dbg !105
  %61 = bitcast void (%swift.context*)* %59 to i8*, !dbg !105
  %62 = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %5, i1 false, void (i8*, %swift.context*)* @__swift_suspend_dispatch_1.2, i8* %61, %swift.context* %60), !dbg !105
  unreachable, !dbg !105
}

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc i8* @swift_task_alloc(i64) #5

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #6

; Function Attrs: nounwind
declare i8* @llvm.coro.async.resume() #3

; Function Attrs: nounwind
define linkonce_odr hidden i8* @__swift_async_resume_get_context(i8* %0) #7 !dbg !106 {
entry:
  ret i8* %0, !dbg !107
}

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_point(i8* %0, i64 %1, i64 %2, %swift.context* %3) #3 !dbg !108 {
entry:
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %3, i8* %0, i64 %1, i64 %2) #3, !dbg !109
  ret void, !dbg !109
}

; Function Attrs: nounwind
declare extern_weak swifttailcc void @swift_task_switch(%swift.context*, i8*, i64, i64) #3

; Function Attrs: nounwind
declare { i8* } @llvm.coro.suspend.async.sl_p0i8s(i32, i8*, i8*, ...) #3

; Function Attrs: alwaysinline nounwind
define linkonce_odr hidden i8* @__swift_async_resume_project_context(i8* %0) #8 !dbg !110 {
entry:
  %1 = bitcast i8* %0 to i8**, !dbg !111
  %2 = load i8*, i8** %1, align 8, !dbg !111
  %3 = call i8** @llvm.swift.async.context.addr(), !dbg !111
  store i8* %2, i8** %3, align 8, !dbg !111
  ret i8* %2, !dbg !111
}

; Function Attrs: nounwind readnone
declare i8** @llvm.swift.async.context.addr() #9

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_1.1(i8* %0, %swift.context* %1) #3 !dbg !112 {
entry:
  %2 = bitcast i8* %0 to void (%swift.context*)*, !dbg !113
  musttail call swifttailcc void %2(%swift.context* swiftasync %1), !dbg !113
  ret void, !dbg !113
}

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc void @swift_task_dealloc(i8*) #5

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #6

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_1.2(i8* %0, %swift.context* %1) #3 !dbg !114 {
entry:
  %2 = bitcast i8* %0 to void (%swift.context*)*, !dbg !115
  musttail call swifttailcc void %2(%swift.context* swiftasync %1), !dbg !115
  ret void, !dbg !115
}

declare extern_weak void @"_swift_FORCE_LOAD_$_swiftCompatibilityConcurrency"()

attributes #0 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nofree nounwind willreturn writeonly }
attributes #3 = { nounwind }
attributes #4 = { cold noreturn nounwind }
attributes #5 = { argmemonly nounwind }
attributes #6 = { argmemonly nofree nosync nounwind willreturn }
attributes #7 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #8 = { alwaysinline nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #9 = { nounwind readnone }

!llvm.dbg.cu = !{!0, !11}
!swift.module.flags = !{!13}
!llvm.module.flags = !{!14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25}
!llvm.linker.options = !{!26, !27, !28, !29, !30}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Swift version 5.7-dev (LLVM 8abcd8862898818, Swift 59a3bd190248a0e)", isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, imports: !2)
!1 = !DIFile(filename: "async_args.swift", directory: "/Volumes/Data/work/solon/build/Ninja+cmark-DebugAssert+llvm-RelWithDebInfoAssert+swift-DebugAssert+stdlib-DebugAssert/swift-macosx-x86_64/tmp/swift")
!2 = !{!3, !5, !7, !9}
!3 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !4, file: !1)
!4 = !DIModule(scope: null, name: "async_args")
!5 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !6, file: !1)
!6 = !DIModule(scope: null, name: "Swift", includePath: "/Volumes/Data/work/solon/build/Ninja+cmark-DebugAssert+llvm-RelWithDebInfoAssert+swift-DebugAssert+stdlib-DebugAssert/swift-macosx-x86_64/lib/swift/macosx/Swift.swiftmodule/x86_64-apple-macos.swiftmodule")
!7 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !8, file: !1)
!8 = !DIModule(scope: null, name: "_Concurrency", includePath: "/Volumes/Data/work/solon/build/Ninja+cmark-DebugAssert+llvm-RelWithDebInfoAssert+swift-DebugAssert+stdlib-DebugAssert/swift-macosx-x86_64/lib/swift/macosx/_Concurrency.swiftmodule/x86_64-apple-macos.swiftmodule")
!9 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !10, file: !1)
!10 = !DIModule(scope: null, name: "SwiftOnoneSupport", includePath: "/Volumes/Data/work/solon/build/Ninja+cmark-DebugAssert+llvm-RelWithDebInfoAssert+swift-DebugAssert+stdlib-DebugAssert/swift-macosx-x86_64/lib/swift/macosx/SwiftOnoneSupport.swiftmodule/x86_64-apple-macos.swiftmodule")
!11 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !12, producer: "clang version 13.0.0 (git@github.com:apple/llvm-project.git 8abcd8862898818152e04399a042997bc185a0e9)", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, sysroot: "/")
!12 = !DIFile(filename: "<swift-imported-modules>", directory: "/Volumes/Data/work/solon/build/Ninja+cmark-DebugAssert+llvm-RelWithDebInfoAssert+swift-DebugAssert+stdlib-DebugAssert/swift-macosx-x86_64/tmp/swift")
!13 = !{!"standard-library", i1 false}
!14 = !{i32 1, !"Objective-C Version", i32 2}
!15 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!16 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!17 = !{i32 4, !"Objective-C Garbage Collection", i32 84346624}
!18 = !{i32 1, !"Objective-C Class Properties", i32 64}
!19 = !{i32 7, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"wchar_size", i32 4}
!22 = !{i32 7, !"PIC Level", i32 2}
!23 = !{i32 7, !"uwtable", i32 1}
!24 = !{i32 7, !"frame-pointer", i32 2}
!25 = !{i32 1, !"Swift Version", i32 7}
!26 = !{!"-lswiftSwiftOnoneSupport"}
!27 = !{!"-lswiftCore"}
!28 = !{!"-lswift_Concurrency"}
!29 = !{!"-lobjc"}
!30 = !{!"-lswiftCompatibilityConcurrency"}
!31 = distinct !DISubprogram(name: "boolean.get", linkageName: "$s10async_args7booleanSbvg", scope: !4, file: !1, line: 6, type: !32, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !36)
!32 = !DISubroutineType(types: !33)
!33 = !{!34}
!34 = !DICompositeType(tag: DW_TAG_structure_type, name: "Bool", scope: !6, file: !35, size: 8, elements: !36, runtimeLang: DW_LANG_Swift, identifier: "$sSbD")
!35 = !DIFile(filename: "lib/swift/macosx/Swift.swiftmodule/x86_64-apple-macos.swiftmodule", directory: "/Volumes/Data/work/solon/build/Ninja+cmark-DebugAssert+llvm-RelWithDebInfoAssert+swift-DebugAssert+stdlib-DebugAssert/swift-macosx-x86_64")
!36 = !{}
!37 = !DILocation(line: 6, column: 27, scope: !38)
!38 = distinct !DILexicalBlock(scope: !31, file: !1, line: 6, column: 19)
!39 = distinct !DISubprogram(name: "use", linkageName: "$s10async_args3useyyxlF", scope: !4, file: !1, line: 8, type: !40, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !44)
!40 = !DISubroutineType(types: !41)
!41 = !{!42, !43}
!42 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sytD", file: !1, elements: !36, runtimeLang: DW_LANG_Swift, identifier: "$sytD")
!43 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sxD", file: !1, runtimeLang: DW_LANG_Swift, identifier: "$sxD")
!44 = !{!45, !50}
!45 = !DILocalVariable(name: "$\CF\84_0_0", scope: !39, file: !1, type: !46, flags: DIFlagArtificial)
!46 = !DIDerivedType(tag: DW_TAG_typedef, name: "T", scope: !48, file: !47, baseType: !49)
!47 = !DIFile(filename: "<compiler-generated>", directory: "")
!48 = !DIModule(scope: null, name: "Builtin")
!49 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "$sBpD", baseType: null, size: 64)
!50 = !DILocalVariable(name: "t", arg: 1, scope: !39, file: !1, line: 8, type: !51)
!51 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !43)
!52 = !DILocation(line: 0, scope: !39)
!53 = !DILocation(line: 8, column: 20, scope: !39)
!54 = !DILocation(line: 8, column: 29, scope: !55)
!55 = distinct !DILexicalBlock(scope: !39, file: !1, line: 8, column: 28)
!56 = distinct !DISubprogram(name: "use2", linkageName: "$s10async_args4use2yyxlF", scope: !4, file: !1, line: 9, type: !40, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !57)
!57 = !{!58, !59}
!58 = !DILocalVariable(name: "$\CF\84_0_0", scope: !56, file: !1, type: !46, flags: DIFlagArtificial)
!59 = !DILocalVariable(name: "t", arg: 1, scope: !56, file: !1, line: 9, type: !51)
!60 = !DILocation(line: 0, scope: !56)
!61 = !DILocation(line: 9, column: 21, scope: !56)
!62 = !DILocation(line: 9, column: 30, scope: !63)
!63 = distinct !DILexicalBlock(scope: !56, file: !1, line: 9, column: 29)
!64 = distinct !DISubprogram(name: "forceSplit", linkageName: "$s10async_args10forceSplityyYaF", scope: !4, file: !1, line: 10, type: !65, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !36)
!65 = !DISubroutineType(types: !66)
!66 = !{!42}
!67 = !DILocation(line: 11, column: 1, scope: !68)
!68 = distinct !DILexicalBlock(scope: !64, file: !1, line: 10, column: 32)
!69 = distinct !DISubprogram(linkageName: "__swift_suspend_dispatch_1", scope: !4, file: !47, type: !70, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !36)
!70 = !DISubroutineType(types: null)
!71 = !DILocation(line: 0, scope: !69)
!72 = distinct !DISubprogram(name: "use3", linkageName: "$s10async_args4use3yyxlF", scope: !4, file: !1, line: 12, type: !40, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !73)
!73 = !{!74, !75}
!74 = !DILocalVariable(name: "$\CF\84_0_0", scope: !72, file: !1, type: !46, flags: DIFlagArtificial)
!75 = !DILocalVariable(name: "t", arg: 1, scope: !72, file: !1, line: 12, type: !51)
!76 = !DILocation(line: 0, scope: !72)
!77 = !DILocation(line: 12, column: 21, scope: !72)
!78 = !DILocation(line: 12, column: 30, scope: !79)
!79 = distinct !DILexicalBlock(scope: !72, file: !1, line: 12, column: 29)
!80 = distinct !DISubprogram(name: "withGenericArg", linkageName: "$s10async_args14withGenericArgyyxnYalF", scope: !4, file: !1, line: 14, type: !40, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !81)
!81 = !{!82, !83, !83, !83}
!82 = !DILocalVariable(name: "$\CF\84_0_0", scope: !80, file: !1, type: !46, flags: DIFlagArtificial)
!83 = !DILocalVariable(name: "msg", arg: 1, scope: !80, file: !1, line: 14, type: !51)
!84 = !DILocation(line: 0, scope: !80)
!85 = !DILocation(line: 0, scope: !86)
!86 = !DILexicalBlockFile(scope: !87, file: !47, discriminator: 0)
!87 = distinct !DILexicalBlock(scope: !80, file: !1, line: 14, column: 55)
!88 = !{i64 96}
!89 = !DILocation(line: 0, scope: !90)
!90 = !DILexicalBlockFile(scope: !80, file: !47, discriminator: 0)
!91 = !DILocation(line: 14, column: 31, scope: !80)
!92 = !DILocation(line: 15, column: 10, scope: !87)
!93 = !DILocation(line: 15, column: 5, scope: !87)
!94 = !DILocation(line: 24, column: 9, scope: !87)
!95 = !DILocation(line: 35, column: 6, scope: !96)
!96 = distinct !DILexicalBlock(scope: !87, file: !1, line: 35, column: 3)
!97 = !DILocation(line: 36, column: 11, scope: !98)
!98 = distinct !DILexicalBlock(scope: !96, file: !1, line: 35, column: 14)
!99 = !DILocation(line: 36, column: 7, scope: !98)
!100 = !DILocation(line: 37, column: 3, scope: !96)
!101 = !DILocation(line: 38, column: 12, scope: !102)
!102 = distinct !DILexicalBlock(scope: !87, file: !1, line: 37, column: 10)
!103 = !DILocation(line: 38, column: 7, scope: !102)
!104 = !DILocation(line: 39, column: 3, scope: !87)
!105 = !DILocation(line: 40, column: 1, scope: !87)
!106 = distinct !DISubprogram(linkageName: "__swift_async_resume_get_context", scope: !4, file: !47, type: !70, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !36)
!107 = !DILocation(line: 0, scope: !106)
!108 = distinct !DISubprogram(linkageName: "__swift_suspend_point", scope: !4, file: !47, type: !70, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !36)
!109 = !DILocation(line: 0, scope: !108)
!110 = distinct !DISubprogram(linkageName: "__swift_async_resume_project_context", scope: !4, file: !47, type: !70, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !36)
!111 = !DILocation(line: 0, scope: !110)
!112 = distinct !DISubprogram(linkageName: "__swift_suspend_dispatch_1.1", scope: !4, file: !47, type: !70, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !36)
!113 = !DILocation(line: 0, scope: !112)
!114 = distinct !DISubprogram(linkageName: "__swift_suspend_dispatch_1.2", scope: !4, file: !47, type: !70, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !36)
!115 = !DILocation(line: 0, scope: !114)
