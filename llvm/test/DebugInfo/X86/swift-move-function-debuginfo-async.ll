;; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
;; RUN: llvm-dwarfdump --show-children %t | FileCheck --check-prefix=DWARF %s

;; REQUIRES: rdar91770227

source_filename = "move_function_dbginfo_async.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.type = type { i64 }
%swift.full_type = type { i8**, %swift.type }
%T27move_function_dbginfo_async5KlassC = type <{ %swift.refcounted }>
%swift.refcounted = type { %swift.type*, i64 }
%swift.opaque = type opaque
%swift.context = type { %swift.context*, void (%swift.context*)* }
%"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame" = type { %swift.context*, %swift.opaque*, %swift.opaque*, %swift.type*, i8**, i8*, i8*, i8* }
%swift.vwtable = type { i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64, i64, i32, i32 }
%"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame" = type { %swift.context*, %swift.opaque*, %swift.opaque*, %swift.opaque*, %swift.opaque*, %swift.type*, i8**, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8* }
%"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame" = type { %swift.context*, %T27move_function_dbginfo_async5KlassC*, %T27move_function_dbginfo_async5KlassC*, %swift.type*, i8*, %T27move_function_dbginfo_async5KlassC*, i8* }
%swift.metadata_response = type { %swift.type*, i64 }
%swift.bridge = type opaque
%Any = type { [24 x i8], %swift.type* }
%TSS = type <{ %Ts11_StringGutsV }>
%Ts11_StringGutsV = type <{ %Ts13_StringObjectV }>
%Ts13_StringObjectV = type <{ %Ts6UInt64V, %swift.bridge* }>
%Ts6UInt64V = type <{ i64 }>

@"$s27move_function_dbginfo_async10forceSplityyYaFTu" = external global %swift.async_func_pointer, align 8
@.str = external hidden unnamed_addr constant [10 x i8]
@"$sSSN" = external global %swift.type, align 8
@"$sypN" = external global %swift.full_type

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare hidden swiftcc %T27move_function_dbginfo_async5KlassC* @"$s27move_function_dbginfo_async5KlassCACycfC"(%swift.type* swiftself) #2

declare swiftcc void @"$s27move_function_dbginfo_async3useyyxlF"(%swift.opaque* noalias nocapture, %swift.type*) #2

declare swifttailcc void @"$s27move_function_dbginfo_async10forceSplityyYaF"(%swift.context* swiftasync) #2

;; DWARF:  DW_AT_linkage_name	("$s27move_function_dbginfo_async13letSimpleTestyyxnYalF")
;; DWARF:  DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x8, DW_OP_deref)
;; DWARF-NEXT:  DW_AT_name ("msg")
define swifttailcc void @"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF"(%swift.context* swiftasync %0, %swift.opaque* noalias %1, %swift.type* %T) #2 !dbg !42 {
entry:
  call void @llvm.dbg.declare(metadata %swift.context* %0, metadata !49, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 24)), !dbg !56
  call void @coro.devirt.trigger(i8* null)
  %T.debug = alloca %swift.type*, align 8
  call void @llvm.dbg.declare(metadata %swift.type** %T.debug, metadata !49, metadata !DIExpression()), !dbg !56
  store %swift.type* %T, %swift.type** %T.debug, align 8
  %2 = bitcast %swift.context* %0 to <{ %swift.context*, void (%swift.context*)* }>*
  %3 = bitcast %swift.context* %0 to i8*
  %async.ctx.frameptr = getelementptr inbounds i8, i8* %3, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr to %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"*
  %4 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 1
  %T.spill.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 3
  store %swift.type* %T, %swift.type** %T.spill.addr, align 8
  %.spill.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 2
  store %swift.opaque* %1, %swift.opaque** %.spill.addr, align 8
  store %swift.context* %0, %swift.context** %4, align 8
  %5 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %5, i8 0, i64 8, i1 false)
  %6 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %6, i8 0, i64 8, i1 false)
  %7 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %7, i8 0, i64 8, i1 false)
  %8 = bitcast %swift.type* %T to i8***, !dbg !57
  %9 = getelementptr inbounds i8**, i8*** %8, i64 -1, !dbg !57
  %T.valueWitnesses = load i8**, i8*** %9, align 8, !dbg !57, !invariant.load !46, !dereferenceable !60
  %T.valueWitnesses.spill.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 4, !dbg !57
  store i8** %T.valueWitnesses, i8*** %T.valueWitnesses.spill.addr, align 8, !dbg !57
  %10 = bitcast i8** %T.valueWitnesses to %swift.vwtable*, !dbg !57
  %11 = getelementptr inbounds %swift.vwtable, %swift.vwtable* %10, i32 0, i32 8, !dbg !57
  %size = load i64, i64* %11, align 8, !dbg !57, !invariant.load !46
  %12 = add i64 %size, 15, !dbg !57
  %13 = and i64 %12, -16, !dbg !57
  %14 = call swiftcc i8* @swift_task_alloc(i64 %13) #7, !dbg !57
  %.spill.addr5 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 5, !dbg !57
  store i8* %14, i8** %.spill.addr5, align 8, !dbg !57
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %14), !dbg !57
  %15 = bitcast i8* %14 to %swift.opaque*, !dbg !57
  %16 = add i64 %size, 15, !dbg !57
  %17 = and i64 %16, -16, !dbg !57
  %18 = call swiftcc i8* @swift_task_alloc(i64 %17) #7, !dbg !57
  %.spill.addr8 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 6, !dbg !57
  store i8* %18, i8** %.spill.addr8, align 8, !dbg !57
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %18), !dbg !57
  %19 = bitcast i8* %18 to %swift.opaque*, !dbg !57
  store %swift.opaque* %1, %swift.opaque** %msg.debug, align 8, !dbg !56
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !61
  call void @llvm.dbg.addr(metadata %swift.context* %0, metadata !54, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !63
  br label %entry.split, !dbg !64

entry.split:                                      ; preds = %entry
  %20 = load i32, i32* getelementptr inbounds (%swift.async_func_pointer, %swift.async_func_pointer* @"$s27move_function_dbginfo_async10forceSplityyYaFTu", i32 0, i32 1), align 8, !dbg !64
  %21 = zext i32 %20 to i64, !dbg !64
  %22 = call swiftcc i8* @swift_task_alloc(i64 %21) #7, !dbg !64
  %.spill.addr11 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 7, !dbg !64
  store i8* %22, i8** %.spill.addr11, align 8, !dbg !64
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %22), !dbg !64
  %23 = bitcast i8* %22 to <{ %swift.context*, void (%swift.context*)* }>*, !dbg !64
  %24 = load %swift.context*, %swift.context** %4, align 8, !dbg !64
  %25 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %23, i32 0, i32 0, !dbg !64
  store %swift.context* %24, %swift.context** %25, align 8, !dbg !64
  %26 = bitcast i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async13letSimpleTestyyxnYalFTQ0_" to i8*) to void (%swift.context*)*, !dbg !64
  %27 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %23, i32 0, i32 1, !dbg !64
  store void (%swift.context*)* %26, void (%swift.context*)** %27, align 8, !dbg !64
  %28 = bitcast i8* %22 to %swift.context*, !dbg !64
  musttail call swifttailcc void @"$s27move_function_dbginfo_async10forceSplityyYaF"(%swift.context* swiftasync %28) #7, !dbg !65
  ret void, !dbg !65
}

;; DWARF:  DW_AT_linkage_name	("$s27move_function_dbginfo_async13letSimpleTestyyxnYalFTQ0_")
;; DWARF:  DW_AT_name	("letSimpleTest")
;; DWARF:  DW_TAG_formal_parameter
;; DWARF-NEXT:  DW_AT_location	(DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_deref, DW_OP_plus_uconst 0x[[MSG_LOC:[a-f0-9]+]], DW_OP_plus_uconst 0x8, DW_OP_deref)
;; DWARF-NEXT:  DW_AT_name	("msg")
define hidden swifttailcc void @"$s27move_function_dbginfo_async13letSimpleTestyyxnYalFTQ0_"(i8* swiftasync %0) #2 !dbg !69 {
entryresume.0:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !71, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 24)), !dbg !73
  %1 = bitcast i8* %0 to i8**, !dbg !74
  %2 = load i8*, i8** %1, align 8, !dbg !74
  %3 = call i8** @llvm.swift.async.context.addr() #7, !dbg !74
  store i8* %2, i8** %3, align 8, !dbg !74
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %2, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr to i8*
  %T.debug = alloca %swift.type*, align 8
  %4 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 1
  %.reload.addr12 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 7, !dbg !78
  %.reload13 = load i8*, i8** %.reload.addr12, align 8, !dbg !78
  %.reload.addr3 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 2, !dbg !78
  %.reload4 = load %swift.opaque*, %swift.opaque** %.reload.addr3, align 8, !dbg !78
  %5 = bitcast i8* %0 to i8**, !dbg !79
  %6 = load i8*, i8** %5, align 8, !dbg !79
  %7 = call i8** @llvm.swift.async.context.addr() #7, !dbg !79
  store i8* %6, i8** %7, align 8, !dbg !79
  %8 = bitcast i8* %6 to %swift.context*, !dbg !78
  store %swift.context* %8, %swift.context** %4, align 8, !dbg !78
  call swiftcc void @swift_task_dealloc(i8* %.reload13) #7, !dbg !78
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %.reload13), !dbg !78
  store %swift.opaque* %.reload4, %swift.opaque** %msg.debug, align 8, !dbg !73
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !81
  call void @llvm.dbg.addr(metadata i8* %0, metadata !72, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !83
  br label %entryresume.0.split, !dbg !84

entryresume.0.split:                              ; preds = %entryresume.0
  %9 = load %swift.context*, %swift.context** %4, align 8, !dbg !84
  %10 = load %swift.context*, %swift.context** %4, align 8, !dbg !84
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %10, i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async13letSimpleTestyyxnYalFTY1_" to i8*), i64 0, i64 0) #7, !dbg !85
  ret void, !dbg !85
}

;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async13letSimpleTestyyxnYalFTY1_")
;; DWARF: DW_AT_name	("letSimpleTest")
;; DWARF: DW_TAG_formal_parameter
;; DWARF: DW_AT_location	(0x{{[a-f0-9]+}}:
;; DWARF-NEXT:            [0x{{[a-f0-9]+}}, 0x{{[a-f0-9]+}}): DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x[[MSG_LOC]], DW_OP_plus_uconst 0x8, DW_OP_deref)
;; DWARF-NEXT:            DW_AT_name	("msg")
define hidden swifttailcc void @"$s27move_function_dbginfo_async13letSimpleTestyyxnYalFTY1_"(i8* swiftasync %0) #2 !dbg !88 {
entryresume.1:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !90, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 24)), !dbg !92
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %0, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr to i8*
  %T.debug = alloca %swift.type*, align 8
  %1 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 1
  %.reload.addr9 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 6, !dbg !93
  %.reload10 = load i8*, i8** %.reload.addr9, align 8, !dbg !93
  %.reload.addr6 = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 5, !dbg !93
  %.reload7 = load i8*, i8** %.reload.addr6, align 8, !dbg !93
  %T.valueWitnesses.reload.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 4, !dbg !93
  %T.valueWitnesses.reload = load i8**, i8*** %T.valueWitnesses.reload.addr, align 8, !dbg !93
  %T.reload.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 3, !dbg !93
  %T.reload = load %swift.type*, %swift.type** %T.reload.addr, align 8, !dbg !93
  %.reload.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame", %"$s27move_function_dbginfo_async13letSimpleTestyyxnYalF.Frame"* %FramePtr, i32 0, i32 2, !dbg !93
  %.reload = load %swift.opaque*, %swift.opaque** %.reload.addr, align 8, !dbg !93
  %2 = bitcast i8* %.reload10 to %swift.opaque*, !dbg !93
  %3 = bitcast i8* %.reload7 to %swift.opaque*, !dbg !93
  %4 = call i8* @__swift_async_resume_get_context(i8* %0), !dbg !96
  %5 = bitcast i8* %4 to %swift.context*, !dbg !96
  store %swift.context* %5, %swift.context** %1, align 8, !dbg !96
  store %swift.opaque* %.reload, %swift.opaque** %msg.debug, align 8, !dbg !92
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !97
  call void @llvm.dbg.addr(metadata i8* %0, metadata !91, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !99
  br label %entryresume.1.split, !dbg !100

entryresume.1.split:                              ; preds = %entryresume.1
  %6 = getelementptr inbounds i8*, i8** %T.valueWitnesses.reload, i32 2, !dbg !100
  %7 = load i8*, i8** %6, align 8, !dbg !100, !invariant.load !46
  %initializeWithCopy = bitcast i8* %7 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !100
  %8 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %3, %swift.opaque* noalias %.reload, %swift.type* %T.reload) #7, !dbg !100
  %9 = getelementptr inbounds i8*, i8** %T.valueWitnesses.reload, i32 4, !dbg !101
  %10 = load i8*, i8** %9, align 8, !dbg !101, !invariant.load !46
  %initializeWithTake = bitcast i8* %10 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !101
  %11 = call %swift.opaque* %initializeWithTake(%swift.opaque* noalias %2, %swift.opaque* noalias %.reload, %swift.type* %T.reload) #7, !dbg !101
  call void @llvm.dbg.value(metadata %swift.opaque* undef, metadata !91, metadata !DIExpression(DW_OP_deref)), !dbg !99
  %12 = getelementptr inbounds i8*, i8** %T.valueWitnesses.reload, i32 1, !dbg !101
  %13 = load i8*, i8** %12, align 8, !dbg !101, !invariant.load !46
  %destroy = bitcast i8* %13 to void (%swift.opaque*, %swift.type*)*, !dbg !101
  call void %destroy(%swift.opaque* noalias %3, %swift.type* %T.reload) #7, !dbg !101
  call swiftcc void @"$s27move_function_dbginfo_async3useyyxlF"(%swift.opaque* noalias nocapture %2, %swift.type* %T.reload), !dbg !102
  call void %destroy(%swift.opaque* noalias %2, %swift.type* %T.reload) #7, !dbg !103
  %14 = bitcast %swift.opaque* %2 to i8*, !dbg !103
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %14), !dbg !103
  call swiftcc void @swift_task_dealloc(i8* %.reload10) #7, !dbg !103
  %15 = bitcast %swift.opaque* %3 to i8*, !dbg !103
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %15), !dbg !103
  call swiftcc void @swift_task_dealloc(i8* %.reload7) #7, !dbg !103
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !103
  %16 = load %swift.context*, %swift.context** %1, align 8, !dbg !103
  %17 = bitcast %swift.context* %16 to <{ %swift.context*, void (%swift.context*)* }>*, !dbg !103
  %18 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %17, i32 0, i32 1, !dbg !103
  %19 = load void (%swift.context*)*, void (%swift.context*)** %18, align 8, !dbg !103
  %20 = load %swift.context*, %swift.context** %1, align 8, !dbg !103
  %21 = bitcast void (%swift.context*)* %19 to i8*, !dbg !103
  musttail call swifttailcc void %19(%swift.context* swiftasync %20) #7, !dbg !104
  ret void, !dbg !104
}

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc i8* @swift_task_alloc(i64) #3

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.addr(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare i8** @llvm.swift.async.context.addr() #5

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc void @swift_task_dealloc(i8*) #3

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: nounwind
declare hidden i8* @__swift_async_resume_get_context(i8*) #6

; Function Attrs: nounwind
declare extern_weak swifttailcc void @swift_task_switch(%swift.context*, i8*, i64, i64) #7

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF")
;; DWARF: DW_AT_name	("varSimpleTest")
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x8, DW_OP_deref)
;; DWARF-NEXT: DW_AT_name ("msg")
define swifttailcc void @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF"(%swift.context* swiftasync %0, %swift.opaque* %1, %swift.opaque* noalias %2, %swift.type* %T) #2 !dbg !107 {
entry:
  call void @llvm.dbg.declare(metadata %swift.context* %0, metadata !113, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16, DW_OP_deref)), !dbg !114
  call void @llvm.dbg.declare(metadata %swift.context* %0, metadata !111, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 40)), !dbg !115
  call void @coro.devirt.trigger(i8* null)
  %T.debug = alloca %swift.type*, align 8
  call void @llvm.dbg.declare(metadata %swift.type** %T.debug, metadata !111, metadata !DIExpression()), !dbg !115
  store %swift.type* %T, %swift.type** %T.debug, align 8
  %3 = bitcast %swift.context* %0 to <{ %swift.context*, void (%swift.context*)* }>*
  %4 = bitcast %swift.context* %0 to i8*
  %async.ctx.frameptr = getelementptr inbounds i8, i8* %4, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr to %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"*
  %5 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 1
  %msg2.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 2
  %T.spill.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 5
  store %swift.type* %T, %swift.type** %T.spill.addr, align 8
  %.spill.addr23 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 4
  store %swift.opaque* %2, %swift.opaque** %.spill.addr23, align 8
  %.spill.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 3
  store %swift.opaque* %1, %swift.opaque** %.spill.addr, align 8
  store %swift.context* %0, %swift.context** %5, align 8
  %6 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %6, i8 0, i64 8, i1 false)
  %7 = bitcast %swift.opaque** %msg2.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %7, i8 0, i64 8, i1 false)
  %8 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %8, i8 0, i64 8, i1 false)
  %9 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %9, i8 0, i64 8, i1 false)
  %10 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %10, i8 0, i64 8, i1 false)
  %11 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %11, i8 0, i64 8, i1 false)
  %12 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %12, i8 0, i64 8, i1 false)
  %13 = bitcast %swift.opaque** %msg.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %13, i8 0, i64 8, i1 false)
  %14 = bitcast %swift.type* %T to i8***, !dbg !116
  %15 = getelementptr inbounds i8**, i8*** %14, i64 -1, !dbg !116
  %T.valueWitnesses = load i8**, i8*** %15, align 8, !dbg !116, !invariant.load !46, !dereferenceable !60
  %T.valueWitnesses.spill.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 6, !dbg !116
  store i8** %T.valueWitnesses, i8*** %T.valueWitnesses.spill.addr, align 8, !dbg !116
  %16 = bitcast i8** %T.valueWitnesses to %swift.vwtable*, !dbg !116
  %17 = getelementptr inbounds %swift.vwtable, %swift.vwtable* %16, i32 0, i32 8, !dbg !116
  %size = load i64, i64* %17, align 8, !dbg !116, !invariant.load !46
  %18 = add i64 %size, 15, !dbg !116
  %19 = and i64 %18, -16, !dbg !116
  %20 = call swiftcc i8* @swift_task_alloc(i64 %19) #7, !dbg !116
  %.spill.addr30 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 7, !dbg !116
  store i8* %20, i8** %.spill.addr30, align 8, !dbg !116
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %20), !dbg !116
  %21 = bitcast i8* %20 to %swift.opaque*, !dbg !116
  %22 = add i64 %size, 15, !dbg !116
  %23 = and i64 %22, -16, !dbg !116
  %24 = call swiftcc i8* @swift_task_alloc(i64 %23) #7, !dbg !116
  %.spill.addr37 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 8, !dbg !116
  store i8* %24, i8** %.spill.addr37, align 8, !dbg !116
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %24), !dbg !116
  %25 = bitcast i8* %24 to %swift.opaque*, !dbg !116
  store %swift.opaque* %1, %swift.opaque** %msg.debug, align 8, !dbg !115
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !119
  call void @llvm.dbg.addr(metadata %swift.context* %0, metadata !112, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !121
  br label %entry.split, !dbg !115

entry.split:                                      ; preds = %entry
  store %swift.opaque* %2, %swift.opaque** %msg2.debug, align 8, !dbg !115
  call void asm sideeffect "", "r"(%swift.opaque** %msg2.debug), !dbg !119
  %26 = load i32, i32* getelementptr inbounds (%swift.async_func_pointer, %swift.async_func_pointer* @"$s27move_function_dbginfo_async10forceSplityyYaFTu", i32 0, i32 1), align 8, !dbg !122
  %27 = zext i32 %26 to i64, !dbg !122
  %28 = call swiftcc i8* @swift_task_alloc(i64 %27) #7, !dbg !122
  %.spill.addr44 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 9, !dbg !122
  store i8* %28, i8** %.spill.addr44, align 8, !dbg !122
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %28), !dbg !122
  %29 = bitcast i8* %28 to <{ %swift.context*, void (%swift.context*)* }>*, !dbg !122
  %30 = load %swift.context*, %swift.context** %5, align 8, !dbg !122
  %31 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %29, i32 0, i32 0, !dbg !122
  store %swift.context* %30, %swift.context** %31, align 8, !dbg !122
  %32 = bitcast i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ0_" to i8*) to void (%swift.context*)*, !dbg !122
  %33 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %29, i32 0, i32 1, !dbg !122
  store void (%swift.context*)* %32, void (%swift.context*)** %33, align 8, !dbg !122
  %34 = bitcast i8* %28 to %swift.context*, !dbg !122
  musttail call swifttailcc void @"$s27move_function_dbginfo_async10forceSplityyYaF"(%swift.context* swiftasync %34) #7, !dbg !123
  ret void, !dbg !123
}

;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ0_")
;; DWARF: DW_AT_name	("varSimpleTest")
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_deref, DW_OP_plus_uconst 0x[[MSG_LOC:[a-f0-9]+]], DW_OP_plus_uconst 0x8, DW_OP_deref)
;; DWARF-NEXT: DW_AT_name	("msg")
define hidden swifttailcc void @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ0_"(i8* swiftasync %0) #2 !dbg !126 {
entryresume.0:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !130, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16, DW_OP_deref)), !dbg !131
  call void @llvm.dbg.declare(metadata i8* %0, metadata !128, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 40)), !dbg !132
  %1 = bitcast i8* %0 to i8**, !dbg !133
  %2 = load i8*, i8** %1, align 8, !dbg !133
  %3 = call i8** @llvm.swift.async.context.addr() #7, !dbg !133
  store i8* %2, i8** %3, align 8, !dbg !133
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %2, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr to i8*
  %T.debug = alloca %swift.type*, align 8
  %4 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 1
  %msg2.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 2
  %.reload.addr45 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 9, !dbg !136
  %.reload46 = load i8*, i8** %.reload.addr45, align 8, !dbg !136
  %.reload.addr21 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 3, !dbg !136
  %.reload22 = load %swift.opaque*, %swift.opaque** %.reload.addr21, align 8, !dbg !136
  %5 = bitcast i8* %0 to i8**, !dbg !137
  %6 = load i8*, i8** %5, align 8, !dbg !137
  %7 = call i8** @llvm.swift.async.context.addr() #7, !dbg !137
  store i8* %6, i8** %7, align 8, !dbg !137
  %8 = bitcast i8* %6 to %swift.context*, !dbg !136
  store %swift.context* %8, %swift.context** %4, align 8, !dbg !136
  call swiftcc void @swift_task_dealloc(i8* %.reload46) #7, !dbg !136
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %.reload46), !dbg !136
  store %swift.opaque* %.reload22, %swift.opaque** %msg.debug, align 8, !dbg !132
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !139
  call void @llvm.dbg.addr(metadata i8* %0, metadata !129, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !141
  br label %entryresume.0.split, !dbg !142

entryresume.0.split:                              ; preds = %entryresume.0
  %9 = load %swift.context*, %swift.context** %4, align 8, !dbg !142
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %9, i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY1_" to i8*), i64 0, i64 0) #7, !dbg !143
  ret void, !dbg !143
}

;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY1_")
;; DWARF: DW_AT_name	("varSimpleTest")
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(0x{{[a-f0-9]+}}:
;; DWARF-NEXT:    [0x{{[a-f0-9]+}}, 0x{{[a-f0-9]+}}): DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x[[MSG_LOC]], DW_OP_plus_uconst 0x8, DW_OP_deref)
;; DWARF-NEXT: DW_AT_name	("msg")
define hidden swifttailcc void @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY1_"(i8* swiftasync %0) #2 !dbg !145 {
entryresume.1:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !149, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16, DW_OP_deref)), !dbg !150
  call void @llvm.dbg.declare(metadata i8* %0, metadata !147, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 40)), !dbg !151
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %0, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr to i8*
  %T.debug = alloca %swift.type*, align 8
  %1 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 1
  %msg2.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 2
  %.reload.addr38 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 8, !dbg !152
  %.reload39 = load i8*, i8** %.reload.addr38, align 8, !dbg !152
  %.reload.addr31 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 7, !dbg !152
  %.reload32 = load i8*, i8** %.reload.addr31, align 8, !dbg !152
  %T.valueWitnesses.reload.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 6, !dbg !152
  %T.valueWitnesses.reload = load i8**, i8*** %T.valueWitnesses.reload.addr, align 8, !dbg !152
  %T.reload.addr28 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 5, !dbg !152
  %T.reload29 = load %swift.type*, %swift.type** %T.reload.addr28, align 8, !dbg !152
  %.reload.addr19 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 3, !dbg !152
  %.reload20 = load %swift.opaque*, %swift.opaque** %.reload.addr19, align 8, !dbg !152
  %2 = bitcast i8* %.reload39 to %swift.opaque*, !dbg !152
  %3 = bitcast i8* %.reload32 to %swift.opaque*, !dbg !152
  %4 = call i8* @__swift_async_resume_get_context(i8* %0), !dbg !155
  %5 = bitcast i8* %4 to %swift.context*, !dbg !155
  store %swift.context* %5, %swift.context** %1, align 8, !dbg !155
  store %swift.opaque* %.reload20, %swift.opaque** %msg.debug, align 8, !dbg !151
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !156
  call void @llvm.dbg.addr(metadata i8* %0, metadata !148, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !158
  br label %entryresume.1.split, !dbg !159

entryresume.1.split:                              ; preds = %entryresume.1
  %6 = getelementptr inbounds i8*, i8** %T.valueWitnesses.reload, i32 2, !dbg !159
  %7 = load i8*, i8** %6, align 8, !dbg !159, !invariant.load !46
  %.spill.addr47 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 10, !dbg !159
  store i8* %7, i8** %.spill.addr47, align 8, !dbg !159
  %initializeWithCopy = bitcast i8* %7 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !159
  %8 = call %swift.opaque* %initializeWithCopy(%swift.opaque* noalias %3, %swift.opaque* noalias %.reload20, %swift.type* %T.reload29) #7, !dbg !159
  %9 = getelementptr inbounds i8*, i8** %T.valueWitnesses.reload, i32 4, !dbg !160
  %10 = load i8*, i8** %9, align 8, !dbg !160, !invariant.load !46
  %.spill.addr50 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 11, !dbg !160
  store i8* %10, i8** %.spill.addr50, align 8, !dbg !160
  %initializeWithTake = bitcast i8* %10 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !160
  %11 = call %swift.opaque* %initializeWithTake(%swift.opaque* noalias %2, %swift.opaque* noalias %.reload20, %swift.type* %T.reload29) #7, !dbg !160
  call void @llvm.dbg.value(metadata %swift.opaque* undef, metadata !148, metadata !DIExpression(DW_OP_deref)), !dbg !158
  %12 = getelementptr inbounds i8*, i8** %T.valueWitnesses.reload, i32 1, !dbg !160
  %13 = load i8*, i8** %12, align 8, !dbg !160, !invariant.load !46
  %.spill.addr53 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 12, !dbg !160
  store i8* %13, i8** %.spill.addr53, align 8, !dbg !160
  %destroy = bitcast i8* %13 to void (%swift.opaque*, %swift.type*)*, !dbg !160
  call void %destroy(%swift.opaque* noalias %3, %swift.type* %T.reload29) #7, !dbg !160
  call swiftcc void @"$s27move_function_dbginfo_async3useyyxlF"(%swift.opaque* noalias nocapture %2, %swift.type* %T.reload29), !dbg !161
  call void %destroy(%swift.opaque* noalias %2, %swift.type* %T.reload29) #7, !dbg !161
  %14 = load i32, i32* getelementptr inbounds (%swift.async_func_pointer, %swift.async_func_pointer* @"$s27move_function_dbginfo_async10forceSplityyYaFTu", i32 0, i32 1), align 8, !dbg !162
  %15 = zext i32 %14 to i64, !dbg !162
  %16 = call swiftcc i8* @swift_task_alloc(i64 %15) #7, !dbg !162
  %.spill.addr58 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 13, !dbg !162
  store i8* %16, i8** %.spill.addr58, align 8, !dbg !162
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %16), !dbg !162
  %17 = bitcast i8* %16 to <{ %swift.context*, void (%swift.context*)* }>*, !dbg !162
  %18 = load %swift.context*, %swift.context** %1, align 8, !dbg !162
  %19 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %17, i32 0, i32 0, !dbg !162
  store %swift.context* %18, %swift.context** %19, align 8, !dbg !162
  %20 = bitcast i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ2_" to i8*) to void (%swift.context*)*, !dbg !162
  %21 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %17, i32 0, i32 1, !dbg !162
  store void (%swift.context*)* %20, void (%swift.context*)** %21, align 8, !dbg !162
  %22 = bitcast i8* %16 to %swift.context*, !dbg !162
  musttail call swifttailcc void @"$s27move_function_dbginfo_async10forceSplityyYaF"(%swift.context* swiftasync %22) #7, !dbg !163
  ret void, !dbg !163
}

;; We were just moved and are not reinit yet. This is caused by us hopping twice
;; when we return from an async function. Once for the async function and then
;; for the hop to executor.
;;
;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ2_")
;; DWARF: DW_AT_name	("varSimpleTest")
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_name ("msg")
define hidden swifttailcc void @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ2_"(i8* swiftasync %0) #2 !dbg !166 {
entryresume.2:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !170, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16, DW_OP_deref)), !dbg !171
  call void @llvm.dbg.declare(metadata i8* %0, metadata !168, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 40)), !dbg !172
  %1 = bitcast i8* %0 to i8**, !dbg !173
  %2 = load i8*, i8** %1, align 8, !dbg !173
  %3 = call i8** @llvm.swift.async.context.addr() #7, !dbg !173
  store i8* %2, i8** %3, align 8, !dbg !173
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %2, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr to i8*
  %T.debug = alloca %swift.type*, align 8
  %4 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 1
  %msg2.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 2
  %.reload.addr59 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 13, !dbg !176
  %.reload60 = load i8*, i8** %.reload.addr59, align 8, !dbg !176
  %5 = bitcast i8* %0 to i8**, !dbg !177
  %6 = load i8*, i8** %5, align 8, !dbg !177
  %7 = call i8** @llvm.swift.async.context.addr() #7, !dbg !177
  store i8* %6, i8** %7, align 8, !dbg !177
  %8 = bitcast i8* %6 to %swift.context*, !dbg !176
  store %swift.context* %8, %swift.context** %4, align 8, !dbg !176
  call swiftcc void @swift_task_dealloc(i8* %.reload60) #7, !dbg !176
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %.reload60), !dbg !176
  call void @llvm.dbg.value(metadata %swift.opaque* undef, metadata !169, metadata !DIExpression(DW_OP_deref)), !dbg !179
  %9 = load %swift.context*, %swift.context** %4, align 8, !dbg !180
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %9, i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY3_" to i8*), i64 0, i64 0) #7, !dbg !181
  ret void, !dbg !181
}

;; We reinitialize our value in this funclet and then move it and then
;; reinitialize it again. So we have two different live ranges. Sadly, we don't
;; validate that the first live range doesn't start at the beginning of the
;; function. But we have lldb tests to validate that.
;;
;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY3_")
;; DWARF: DW_AT_name	("varSimpleTest")
;; DWARF: DW_TAG_formal_parameter
;; DWARF: DW_AT_location	(0x{{[a-f0-9]+}}:
;; DWARF-NEXT:    [0x{{[a-f0-9]+}}, 0x{{[a-f0-9]+}}):
;; DWARF-SAME:        DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x[[MSG_LOC]], DW_OP_plus_uconst 0x8, DW_OP_deref
;; DWARF-NEXT:    [0x{{[a-f0-9]+}}, 0x{{[a-f0-9]+}}):
;; DWARF-SAME:        DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x[[MSG_LOC]], DW_OP_plus_uconst 0x8, DW_OP_deref
;; DWARF-NEXT: DW_AT_name	("msg")
define hidden swifttailcc void @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY3_"(i8* swiftasync %0) #2 !dbg !183 {
entryresume.3:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !187, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16, DW_OP_deref)), !dbg !188
  call void @llvm.dbg.declare(metadata i8* %0, metadata !185, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 40)), !dbg !189
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %0, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr to i8*
  %T.debug = alloca %swift.type*, align 8
  %1 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 1
  %msg2.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 2
  %.reload.addr54 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 12, !dbg !190
  %.reload55 = load i8*, i8** %.reload.addr54, align 8, !dbg !190
  %.reload.addr51 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 11, !dbg !190
  %.reload52 = load i8*, i8** %.reload.addr51, align 8, !dbg !190
  %.reload.addr48 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 10, !dbg !190
  %.reload49 = load i8*, i8** %.reload.addr48, align 8, !dbg !190
  %.reload.addr40 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 8, !dbg !190
  %.reload41 = load i8*, i8** %.reload.addr40, align 8, !dbg !190
  %.reload.addr33 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 7, !dbg !190
  %.reload34 = load i8*, i8** %.reload.addr33, align 8, !dbg !190
  %T.reload.addr26 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 5, !dbg !190
  %T.reload27 = load %swift.type*, %swift.type** %T.reload.addr26, align 8, !dbg !190
  %.reload.addr24 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 4, !dbg !190
  %.reload25 = load %swift.opaque*, %swift.opaque** %.reload.addr24, align 8, !dbg !190
  %.reload.addr17 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 3, !dbg !190
  %.reload18 = load %swift.opaque*, %swift.opaque** %.reload.addr17, align 8, !dbg !190
  %destroy14 = bitcast i8* %.reload55 to void (%swift.opaque*, %swift.type*)*, !dbg !190
  %initializeWithTake12 = bitcast i8* %.reload52 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !190
  %initializeWithCopy11 = bitcast i8* %.reload49 to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*, !dbg !192
  %2 = bitcast i8* %.reload41 to %swift.opaque*, !dbg !193
  %3 = bitcast i8* %.reload34 to %swift.opaque*, !dbg !193
  %4 = call i8* @__swift_async_resume_get_context(i8* %0), !dbg !195
  %5 = bitcast i8* %4 to %swift.context*, !dbg !195
  store %swift.context* %5, %swift.context** %1, align 8, !dbg !195
  call void @llvm.dbg.value(metadata %swift.opaque* undef, metadata !186, metadata !DIExpression(DW_OP_deref)), !dbg !196
  %6 = call %swift.opaque* %initializeWithCopy11(%swift.opaque* noalias %2, %swift.opaque* noalias %.reload25, %swift.type* %T.reload27) #7, !dbg !197
  %7 = call %swift.opaque* %initializeWithTake12(%swift.opaque* noalias %.reload18, %swift.opaque* noalias %2, %swift.type* %T.reload27) #7, !dbg !198
  store %swift.opaque* %.reload18, %swift.opaque** %msg.debug, align 8, !dbg !189
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !199
  call void @llvm.dbg.addr(metadata i8* %0, metadata !186, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !196
  br label %entryresume.3.split2, !dbg !201

entryresume.3.split2:                             ; preds = %entryresume.3
  %8 = call %swift.opaque* %initializeWithCopy11(%swift.opaque* noalias %3, %swift.opaque* noalias %.reload18, %swift.type* %T.reload27) #7, !dbg !201
  %9 = call %swift.opaque* %initializeWithTake12(%swift.opaque* noalias %2, %swift.opaque* noalias %.reload18, %swift.type* %T.reload27) #7, !dbg !202
  call void @llvm.dbg.value(metadata %swift.opaque* undef, metadata !186, metadata !DIExpression(DW_OP_deref)), !dbg !196
  call void %destroy14(%swift.opaque* noalias %3, %swift.type* %T.reload27) #7, !dbg !202
  %10 = call %swift.opaque* %initializeWithCopy11(%swift.opaque* noalias %3, %swift.opaque* noalias %2, %swift.type* %T.reload27) #7, !dbg !203
  call void %destroy14(%swift.opaque* noalias %3, %swift.type* %T.reload27) #7, !dbg !203
  %11 = call %swift.opaque* %initializeWithCopy11(%swift.opaque* noalias %3, %swift.opaque* noalias %.reload25, %swift.type* %T.reload27) #7, !dbg !204
  %12 = call %swift.opaque* %initializeWithTake12(%swift.opaque* noalias %.reload18, %swift.opaque* noalias %3, %swift.type* %T.reload27) #7, !dbg !205
  store %swift.opaque* %.reload18, %swift.opaque** %msg.debug, align 8, !dbg !189
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !199
  call void @llvm.dbg.addr(metadata i8* %0, metadata !186, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !196
  br label %entryresume.3.split, !dbg !206

entryresume.3.split:                              ; preds = %entryresume.3.split2
  %13 = load i32, i32* getelementptr inbounds (%swift.async_func_pointer, %swift.async_func_pointer* @"$s27move_function_dbginfo_async10forceSplityyYaFTu", i32 0, i32 1), align 8, !dbg !206
  %14 = zext i32 %13 to i64, !dbg !206
  %15 = call swiftcc i8* @swift_task_alloc(i64 %14) #7, !dbg !206
  %.spill.addr61 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 14, !dbg !206
  store i8* %15, i8** %.spill.addr61, align 8, !dbg !206
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %15), !dbg !206
  %16 = bitcast i8* %15 to <{ %swift.context*, void (%swift.context*)* }>*, !dbg !206
  %17 = load %swift.context*, %swift.context** %1, align 8, !dbg !206
  %18 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %16, i32 0, i32 0, !dbg !206
  store %swift.context* %17, %swift.context** %18, align 8, !dbg !206
  %19 = bitcast i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ4_" to i8*) to void (%swift.context*)*, !dbg !206
  %20 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %16, i32 0, i32 1, !dbg !206
  store void (%swift.context*)* %19, void (%swift.context*)** %20, align 8, !dbg !206
  %21 = bitcast i8* %15 to %swift.context*, !dbg !206
  musttail call swifttailcc void @"$s27move_function_dbginfo_async10forceSplityyYaF"(%swift.context* swiftasync %21) #7, !dbg !207
  ret void, !dbg !207
}

;; We did not move the value again here, so we just get a normal entry value for
;; the entire function.
;;
;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ4_")
;; DWARF: DW_AT_name	("varSimpleTest")
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_deref, DW_OP_plus_uconst 0x[[MSG_LOC]], DW_OP_plus_uconst 0x8, DW_OP_deref)
;; DWARF-NEXT: DW_AT_name	("msg")
define hidden swifttailcc void @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ4_"(i8* swiftasync %0) #2 !dbg !210 {
entryresume.4:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !214, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16, DW_OP_deref)), !dbg !215
  call void @llvm.dbg.declare(metadata i8* %0, metadata !212, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 40)), !dbg !216
  %1 = bitcast i8* %0 to i8**, !dbg !217
  %2 = load i8*, i8** %1, align 8, !dbg !217
  %3 = call i8** @llvm.swift.async.context.addr() #7, !dbg !217
  store i8* %2, i8** %3, align 8, !dbg !217
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %2, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr to i8*
  %T.debug = alloca %swift.type*, align 8
  %4 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 1
  %msg2.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 2
  %.reload.addr62 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 14, !dbg !220
  %.reload63 = load i8*, i8** %.reload.addr62, align 8, !dbg !220
  %.reload.addr15 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 3, !dbg !220
  %.reload16 = load %swift.opaque*, %swift.opaque** %.reload.addr15, align 8, !dbg !220
  %5 = bitcast i8* %0 to i8**, !dbg !221
  %6 = load i8*, i8** %5, align 8, !dbg !221
  %7 = call i8** @llvm.swift.async.context.addr() #7, !dbg !221
  store i8* %6, i8** %7, align 8, !dbg !221
  %8 = bitcast i8* %6 to %swift.context*, !dbg !220
  store %swift.context* %8, %swift.context** %4, align 8, !dbg !220
  call swiftcc void @swift_task_dealloc(i8* %.reload63) #7, !dbg !220
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %.reload63), !dbg !220
  store %swift.opaque* %.reload16, %swift.opaque** %msg.debug, align 8, !dbg !216
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !223
  call void @llvm.dbg.addr(metadata i8* %0, metadata !213, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !225
  br label %entryresume.4.split, !dbg !226

entryresume.4.split:                              ; preds = %entryresume.4
  %9 = load %swift.context*, %swift.context** %4, align 8, !dbg !226
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %9, i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY5_" to i8*), i64 0, i64 0) #7, !dbg !227
  ret void, !dbg !227
}

;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY5_")
;; DWARF: DW_AT_name	("varSimpleTest")
;; DWARF: DW_TAG_formal_parameter
;; DWARF-NEXT: DW_AT_location	(DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x8, DW_OP_deref
;; DWARF-NEXT: DW_AT_name	("msg")
define hidden swifttailcc void @"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY5_"(i8* swiftasync %0) #2 !dbg !229 {
entryresume.5:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !233, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16, DW_OP_deref)), !dbg !234
  call void @llvm.dbg.declare(metadata i8* %0, metadata !231, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 40)), !dbg !235
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %0, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr to i8*
  %T.debug = alloca %swift.type*, align 8
  %1 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 0
  %msg.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 1
  %msg2.debug = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 2
  %.reload.addr56 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 12, !dbg !236
  %.reload57 = load i8*, i8** %.reload.addr56, align 8, !dbg !236
  %.reload.addr42 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 8, !dbg !236
  %.reload43 = load i8*, i8** %.reload.addr42, align 8, !dbg !236
  %.reload.addr35 = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 7, !dbg !236
  %.reload36 = load i8*, i8** %.reload.addr35, align 8, !dbg !236
  %T.reload.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 5, !dbg !236
  %T.reload = load %swift.type*, %swift.type** %T.reload.addr, align 8, !dbg !236
  %.reload.addr = getelementptr inbounds %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame", %"$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF.Frame"* %FramePtr, i32 0, i32 3, !dbg !236
  %.reload = load %swift.opaque*, %swift.opaque** %.reload.addr, align 8, !dbg !236
  %destroy13 = bitcast i8* %.reload57 to void (%swift.opaque*, %swift.type*)*, !dbg !236
  %2 = bitcast i8* %.reload43 to %swift.opaque*, !dbg !238
  %3 = bitcast i8* %.reload36 to %swift.opaque*, !dbg !238
  %4 = call i8* @__swift_async_resume_get_context(i8* %0), !dbg !240
  %5 = bitcast i8* %4 to %swift.context*, !dbg !240
  store %swift.context* %5, %swift.context** %1, align 8, !dbg !240
  store %swift.opaque* %.reload, %swift.opaque** %msg.debug, align 8, !dbg !235
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !241
  call void @llvm.dbg.addr(metadata i8* %0, metadata !232, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8, DW_OP_deref)), !dbg !243
  br label %entryresume.5.split, !dbg !244

entryresume.5.split:                              ; preds = %entryresume.5
  call void %destroy13(%swift.opaque* noalias %2, %swift.type* %T.reload) #7, !dbg !244
  %6 = bitcast %swift.opaque* %2 to i8*, !dbg !244
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %6), !dbg !244
  call swiftcc void @swift_task_dealloc(i8* %.reload43) #7, !dbg !244
  %7 = bitcast %swift.opaque* %3 to i8*, !dbg !244
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7), !dbg !244
  call swiftcc void @swift_task_dealloc(i8* %.reload36) #7, !dbg !244
  call void asm sideeffect "", "r"(%swift.opaque** %msg.debug), !dbg !244
  call void asm sideeffect "", "r"(%swift.opaque** %msg2.debug), !dbg !244
  %8 = load %swift.context*, %swift.context** %1, align 8, !dbg !244
  %9 = bitcast %swift.context* %8 to <{ %swift.context*, void (%swift.context*)* }>*, !dbg !244
  %10 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %9, i32 0, i32 1, !dbg !244
  %11 = load void (%swift.context*)*, void (%swift.context*)** %10, align 8, !dbg !244
  %12 = load %swift.context*, %swift.context** %1, align 8, !dbg !244
  %13 = bitcast void (%swift.context*)* %11 to i8*, !dbg !244
  musttail call swifttailcc void %11(%swift.context* swiftasync %12) #7, !dbg !245
  ret void, !dbg !245
}

;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async16varSimpleTestVaryyYaF")
;: DWARF-NOT: DW_AT_name ("k")
define swifttailcc void @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF"(%swift.context* swiftasync %0) #2 !dbg !248 {
entry:
  call void @llvm.dbg.declare(metadata %swift.context* %0, metadata !255, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)), !dbg !257
  call void @coro.devirt.trigger(i8* null)
  %1 = bitcast %swift.context* %0 to <{ %swift.context*, void (%swift.context*)* }>*
  %2 = bitcast %swift.context* %0 to i8*
  %async.ctx.frameptr = getelementptr inbounds i8, i8* %2, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr to %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"*
  %3 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 0
  %k = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 1
  %m.debug = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 2
  %4 = load %T27move_function_dbginfo_async5KlassC*, %T27move_function_dbginfo_async5KlassC** %k, align 8
  store %swift.context* %0, %swift.context** %3, align 8
  %5 = bitcast %T27move_function_dbginfo_async5KlassC** %k to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %5, i8 0, i64 8, i1 false)
  %6 = bitcast %T27move_function_dbginfo_async5KlassC** %m.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %6, i8 0, i64 8, i1 false)
  %7 = load %swift.context*, %swift.context** %3, align 8, !dbg !258
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %7, i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY0_" to i8*), i64 0, i64 0) #7, !dbg !259
  ret void, !dbg !259
}

;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY0_")
;;
;; DWARF:    DW_TAG_variable
;; DWARF-NEXT: DW_AT_location   (DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x8)
;; DWARF-NEXT: DW_AT_name       ("k")
;;
;; DWARF:    DW_TAG_variable
;; DWARF-NEXT: DW_AT_location
;; DWARF-NEXT: DW_AT_name ("m")
define hidden swifttailcc void @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY0_"(i8* swiftasync %0) #2 !dbg !261 {
entryresume.0:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !265, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)), !dbg !266
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %0, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr to i8*
  %1 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 0
  %k = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 1
  %m.debug = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 2
  %2 = call i8* @__swift_async_resume_get_context(i8* %0), !dbg !267
  %3 = bitcast i8* %2 to %swift.context*, !dbg !267
  store %swift.context* %3, %swift.context** %1, align 8, !dbg !267
  %4 = bitcast %T27move_function_dbginfo_async5KlassC** %k to i8*, !dbg !268
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %4), !dbg !268
  call void asm sideeffect "", "r"(%T27move_function_dbginfo_async5KlassC** %k), !dbg !268
  call void @llvm.dbg.addr(metadata i8* %0, metadata !263, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8)), !dbg !270
  br label %entryresume.0.split, !dbg !271

entryresume.0.split:                              ; preds = %entryresume.0
  %5 = call swiftcc %swift.metadata_response @"$s27move_function_dbginfo_async5KlassCMa"(i64 0) #5, !dbg !271
  %6 = extractvalue %swift.metadata_response %5, 0, !dbg !271
  %.spill.addr = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 3, !dbg !271
  store %swift.type* %6, %swift.type** %.spill.addr, align 8, !dbg !271
  %7 = call swiftcc %T27move_function_dbginfo_async5KlassC* @"$s27move_function_dbginfo_async5KlassCACycfC"(%swift.type* swiftself %6), !dbg !271
  %8 = bitcast %T27move_function_dbginfo_async5KlassC* %7 to %swift.refcounted*, !dbg !271
  %9 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %8) #7, !dbg !271
  store %T27move_function_dbginfo_async5KlassC* %7, %T27move_function_dbginfo_async5KlassC** %k, align 8, !dbg !271
  %10 = getelementptr inbounds %T27move_function_dbginfo_async5KlassC, %T27move_function_dbginfo_async5KlassC* %7, i32 0, i32 0, i32 0, !dbg !272
  %11 = load %swift.type*, %swift.type** %10, align 8, !dbg !272
  %12 = bitcast %swift.type* %11 to void (%T27move_function_dbginfo_async5KlassC*)**, !dbg !272
  %13 = getelementptr inbounds void (%T27move_function_dbginfo_async5KlassC*)*, void (%T27move_function_dbginfo_async5KlassC*)** %12, i64 10, !dbg !272
  %14 = load void (%T27move_function_dbginfo_async5KlassC*)*, void (%T27move_function_dbginfo_async5KlassC*)** %13, align 8, !dbg !272, !invariant.load !46
  call swiftcc void %14(%T27move_function_dbginfo_async5KlassC* swiftself %7), !dbg !272
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T27move_function_dbginfo_async5KlassC*)*)(%T27move_function_dbginfo_async5KlassC* %7) #7, !dbg !272
  %15 = load i32, i32* getelementptr inbounds (%swift.async_func_pointer, %swift.async_func_pointer* @"$s27move_function_dbginfo_async10forceSplityyYaFTu", i32 0, i32 1), align 8, !dbg !273
  %16 = zext i32 %15 to i64, !dbg !273
  %17 = call swiftcc i8* @swift_task_alloc(i64 %16) #7, !dbg !273
  %.spill.addr9 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 4, !dbg !273
  store i8* %17, i8** %.spill.addr9, align 8, !dbg !273
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %17), !dbg !273
  %18 = bitcast i8* %17 to <{ %swift.context*, void (%swift.context*)* }>*, !dbg !273
  %19 = load %swift.context*, %swift.context** %1, align 8, !dbg !273
  %20 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %18, i32 0, i32 0, !dbg !273
  store %swift.context* %19, %swift.context** %20, align 8, !dbg !273
  %21 = bitcast i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTQ1_" to i8*) to void (%swift.context*)*, !dbg !273
  %22 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %18, i32 0, i32 1, !dbg !273
  store void (%swift.context*)* %21, void (%swift.context*)** %22, align 8, !dbg !273
  %23 = bitcast i8* %17 to %swift.context*, !dbg !273
  musttail call swifttailcc void @"$s27move_function_dbginfo_async10forceSplityyYaF"(%swift.context* swiftasync %23) #7, !dbg !274
  ret void, !dbg !274
}

;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTQ1_")
;;
;; DWARF:    DW_TAG_variable
;; DWARF-NEXT: DW_AT_location   (DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_deref, DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x8)
;; DWARF-NEXT: DW_AT_name       ("k")
;;
;; DWARF:    DW_TAG_variable
;; DWARF-NEXT: DW_AT_location	(DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_deref, DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x10)
;; DWARF-NEXT: DW_AT_name ("m")
define hidden swifttailcc void @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTQ1_"(i8* swiftasync %0) #2 !dbg !277 {
entryresume.1:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !281, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)), !dbg !282
  %1 = bitcast i8* %0 to i8**, !dbg !283
  %2 = load i8*, i8** %1, align 8, !dbg !283
  %3 = call i8** @llvm.swift.async.context.addr() #7, !dbg !283
  store i8* %2, i8** %3, align 8, !dbg !283
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %2, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr to i8*
  %4 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 0
  %k = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 1
  %m.debug = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 2
  %.reload.addr10 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 4, !dbg !285
  %.reload11 = load i8*, i8** %.reload.addr10, align 8, !dbg !285
  %5 = bitcast i8* %0 to i8**, !dbg !286
  %6 = load i8*, i8** %5, align 8, !dbg !286
  %7 = call i8** @llvm.swift.async.context.addr() #7, !dbg !286
  store i8* %6, i8** %7, align 8, !dbg !286
  %8 = bitcast i8* %6 to %swift.context*, !dbg !285
  store %swift.context* %8, %swift.context** %4, align 8, !dbg !285
  call swiftcc void @swift_task_dealloc(i8* %.reload11) #7, !dbg !285
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %.reload11), !dbg !285
  call void asm sideeffect "", "r"(%T27move_function_dbginfo_async5KlassC** %k), !dbg !288
  call void @llvm.dbg.addr(metadata i8* %0, metadata !279, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8)), !dbg !290
  br label %entryresume.1.split, !dbg !288

entryresume.1.split:                              ; preds = %entryresume.1
  %9 = load %swift.context*, %swift.context** %4, align 8, !dbg !288
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %9, i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY2_" to i8*), i64 0, i64 0) #7, !dbg !291
  ret void, !dbg !291
}

;; DWARF: DW_AT_linkage_name	("$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY2_")
;; DWARF:    DW_TAG_variable
;; DWARF-NEXT: DW_AT_location   (0x{{[0-9a-f]+}}:
;; DWARF-NEXT:    [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x8)
;; DWARF-NEXT: DW_AT_name       ("k")
;; DWARF:    DW_TAG_variable
;; DWARF-NEXT: DW_AT_location
;; DWARF-NEXT: DW_AT_name ("m")
define hidden swifttailcc void @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY2_"(i8* swiftasync %0) #2 !dbg !293 {
entryresume.2:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !297, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)), !dbg !298
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %0, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr to i8*
  %1 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 0
  %k = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 1
  %m.debug = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 2
  %2 = call i8* @__swift_async_resume_get_context(i8* %0), !dbg !299
  %3 = bitcast i8* %2 to %swift.context*, !dbg !299
  store %swift.context* %3, %swift.context** %1, align 8, !dbg !299
  call void asm sideeffect "", "r"(%T27move_function_dbginfo_async5KlassC** %k), !dbg !299
  call void @llvm.dbg.addr(metadata i8* %0, metadata !295, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8)), !dbg !301
  br label %entryresume.2.split, !dbg !302

entryresume.2.split:                              ; preds = %entryresume.2
  %4 = load %T27move_function_dbginfo_async5KlassC*, %T27move_function_dbginfo_async5KlassC** %k, align 8, !dbg !302
  %.spill.addr12 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 5, !dbg !301
  store %T27move_function_dbginfo_async5KlassC* %4, %T27move_function_dbginfo_async5KlassC** %.spill.addr12, align 8, !dbg !301
  call void @llvm.dbg.value(metadata %T27move_function_dbginfo_async5KlassC** undef, metadata !295, metadata !DIExpression()), !dbg !301
  store %T27move_function_dbginfo_async5KlassC* %4, %T27move_function_dbginfo_async5KlassC** %m.debug, align 8, !dbg !303
  call void asm sideeffect "", "r"(%T27move_function_dbginfo_async5KlassC** %m.debug), !dbg !299
  %5 = getelementptr inbounds %T27move_function_dbginfo_async5KlassC, %T27move_function_dbginfo_async5KlassC* %4, i32 0, i32 0, i32 0, !dbg !304
  %6 = load %swift.type*, %swift.type** %5, align 8, !dbg !304
  %7 = bitcast %swift.type* %6 to void (%T27move_function_dbginfo_async5KlassC*)**, !dbg !304
  %8 = getelementptr inbounds void (%T27move_function_dbginfo_async5KlassC*)*, void (%T27move_function_dbginfo_async5KlassC*)** %7, i64 10, !dbg !304
  %9 = load void (%T27move_function_dbginfo_async5KlassC*)*, void (%T27move_function_dbginfo_async5KlassC*)** %8, align 8, !dbg !304, !invariant.load !46
  call swiftcc void %9(%T27move_function_dbginfo_async5KlassC* swiftself %4), !dbg !304
  %10 = load i32, i32* getelementptr inbounds (%swift.async_func_pointer, %swift.async_func_pointer* @"$s27move_function_dbginfo_async10forceSplityyYaFTu", i32 0, i32 1), align 8, !dbg !305
  %11 = zext i32 %10 to i64, !dbg !305
  %12 = call swiftcc i8* @swift_task_alloc(i64 %11) #7, !dbg !305
  %.spill.addr15 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 6, !dbg !305
  store i8* %12, i8** %.spill.addr15, align 8, !dbg !305
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %12), !dbg !305
  %13 = bitcast i8* %12 to <{ %swift.context*, void (%swift.context*)* }>*, !dbg !305
  %14 = load %swift.context*, %swift.context** %1, align 8, !dbg !305
  %15 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %13, i32 0, i32 0, !dbg !305
  store %swift.context* %14, %swift.context** %15, align 8, !dbg !305
  %16 = bitcast i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTQ3_" to i8*) to void (%swift.context*)*, !dbg !305
  %17 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %13, i32 0, i32 1, !dbg !305
  store void (%swift.context*)* %16, void (%swift.context*)** %17, align 8, !dbg !305
  %18 = bitcast i8* %12 to %swift.context*, !dbg !305
  musttail call swifttailcc void @"$s27move_function_dbginfo_async10forceSplityyYaF"(%swift.context* swiftasync %18) #7, !dbg !306
  ret void, !dbg !306
}

;; DWARF: DW_AT_linkage_name  ("$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTQ3_")
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location  (DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_deref, DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x10)
;; DWARF-NEXT: DW_AT_name  ("m")
;; K is dead here.
;; DWARF: DW_TAG_variable
;; DWARF-NEXT:    DW_AT_name  ("k")
define hidden swifttailcc void @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTQ3_"(i8* swiftasync %0) #2 !dbg !309 {
entryresume.3:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !313, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)), !dbg !314
  %1 = bitcast i8* %0 to i8**, !dbg !315
  %2 = load i8*, i8** %1, align 8, !dbg !315
  %3 = call i8** @llvm.swift.async.context.addr() #7, !dbg !315
  store i8* %2, i8** %3, align 8, !dbg !315
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %2, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr to i8*
  %4 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 0
  %k = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 1
  %m.debug = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 2
  %.reload.addr16 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 6, !dbg !317
  %.reload17 = load i8*, i8** %.reload.addr16, align 8, !dbg !317
  %5 = bitcast i8* %0 to i8**, !dbg !318
  %6 = load i8*, i8** %5, align 8, !dbg !318
  %7 = call i8** @llvm.swift.async.context.addr() #7, !dbg !318
  store i8* %6, i8** %7, align 8, !dbg !318
  %8 = bitcast i8* %6 to %swift.context*, !dbg !317
  store %swift.context* %8, %swift.context** %4, align 8, !dbg !317
  call swiftcc void @swift_task_dealloc(i8* %.reload17) #7, !dbg !317
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %.reload17), !dbg !317
  call void @llvm.dbg.value(metadata %T27move_function_dbginfo_async5KlassC** undef, metadata !311, metadata !DIExpression()), !dbg !320
  %9 = load %swift.context*, %swift.context** %4, align 8, !dbg !321
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %9, i8* bitcast (void (i8*)* @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY4_" to i8*), i64 0, i64 0) #7, !dbg !323
  ret void, !dbg !323
}

;; We reinitialize k in 4.
;; DWARF: DW_AT_linkage_name  ("$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY4_")
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location  (0x{{[0-9a-f]+}}:
;; DWARF-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x8)
;; DWARF-NEXT: DW_AT_name ("k")
;; DWARF: DW_TAG_variable
;; DWARF-NEXT: DW_AT_location  (DW_OP_entry_value(DW_OP_reg14 R14), DW_OP_plus_uconst 0x10, DW_OP_plus_uconst 0x10)
;; DWARF-NEXT: DW_AT_name  ("m")
define hidden swifttailcc void @"$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY4_"(i8* swiftasync %0) #2 !dbg !325 {
entryresume.4:
  call void @llvm.dbg.declare(metadata i8* %0, metadata !329, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 16)), !dbg !330
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %0, i32 16
  %FramePtr = bitcast i8* %async.ctx.frameptr1 to %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"*
  %vFrame = bitcast %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr to i8*
  %1 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 0
  %k = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 1
  %m.debug = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 2
  %.reload.addr13 = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 5, !dbg !331
  %.reload14 = load %T27move_function_dbginfo_async5KlassC*, %T27move_function_dbginfo_async5KlassC** %.reload.addr13, align 8, !dbg !331
  %.reload.addr = getelementptr inbounds %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame", %"$s27move_function_dbginfo_async16varSimpleTestVaryyYaF.Frame"* %FramePtr, i32 0, i32 3, !dbg !331
  %.reload = load %swift.type*, %swift.type** %.reload.addr, align 8, !dbg !331
  %2 = call i8* @__swift_async_resume_get_context(i8* %0), !dbg !331
  %3 = bitcast i8* %2 to %swift.context*, !dbg !331
  store %swift.context* %3, %swift.context** %1, align 8, !dbg !331
  call void @llvm.dbg.value(metadata %T27move_function_dbginfo_async5KlassC** undef, metadata !327, metadata !DIExpression()), !dbg !333
  %4 = call swiftcc %T27move_function_dbginfo_async5KlassC* @"$s27move_function_dbginfo_async5KlassCACycfC"(%swift.type* swiftself %.reload), !dbg !334
  %5 = bitcast %T27move_function_dbginfo_async5KlassC* %4 to %swift.refcounted*, !dbg !335
  %6 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %5) #7, !dbg !335
  store %T27move_function_dbginfo_async5KlassC* %4, %T27move_function_dbginfo_async5KlassC** %k, align 8, !dbg !335
  call void asm sideeffect "", "r"(%T27move_function_dbginfo_async5KlassC** %k), !dbg !331
  call void @llvm.dbg.addr(metadata i8* %0, metadata !327, metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_plus_uconst, 8)), !dbg !333
  br label %entryresume.4.split, !dbg !336

entryresume.4.split:                              ; preds = %entryresume.4
  %7 = getelementptr inbounds %T27move_function_dbginfo_async5KlassC, %T27move_function_dbginfo_async5KlassC* %4, i32 0, i32 0, i32 0, !dbg !336
  %8 = load %swift.type*, %swift.type** %7, align 8, !dbg !336
  %9 = bitcast %swift.type* %8 to void (%T27move_function_dbginfo_async5KlassC*)**, !dbg !336
  %10 = getelementptr inbounds void (%T27move_function_dbginfo_async5KlassC*)*, void (%T27move_function_dbginfo_async5KlassC*)** %9, i64 10, !dbg !336
  %11 = load void (%T27move_function_dbginfo_async5KlassC*)*, void (%T27move_function_dbginfo_async5KlassC*)** %10, align 8, !dbg !336, !invariant.load !46
  call swiftcc void %11(%T27move_function_dbginfo_async5KlassC* swiftself %4), !dbg !336
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T27move_function_dbginfo_async5KlassC*)*)(%T27move_function_dbginfo_async5KlassC* %4) #7, !dbg !336
  %12 = call swiftcc { %swift.bridge*, i8* } @"$ss27_allocateUninitializedArrayySayxG_BptBwlFyp_Tg5"(i64 1), !dbg !337
  %13 = extractvalue { %swift.bridge*, i8* } %12, 0, !dbg !337
  %14 = extractvalue { %swift.bridge*, i8* } %12, 1, !dbg !337
  %15 = bitcast i8* %14 to %Any*, !dbg !337
  %16 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i64 0, i64 0), i64 9, i1 true), !dbg !337
  %17 = extractvalue { i64, %swift.bridge* } %16, 0, !dbg !337
  %18 = extractvalue { i64, %swift.bridge* } %16, 1, !dbg !337
  %19 = getelementptr inbounds %Any, %Any* %15, i32 0, i32 1, !dbg !337
  store %swift.type* @"$sSSN", %swift.type** %19, align 8, !dbg !337
  %20 = getelementptr inbounds %Any, %Any* %15, i32 0, i32 0, !dbg !337
  %21 = getelementptr inbounds %Any, %Any* %15, i32 0, i32 0, !dbg !337
  %22 = bitcast [24 x i8]* %21 to %TSS*, !dbg !337
  %._guts = getelementptr inbounds %TSS, %TSS* %22, i32 0, i32 0, !dbg !337
  %._guts._object = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %._guts, i32 0, i32 0, !dbg !337
  %._guts._object._countAndFlagsBits = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %._guts._object, i32 0, i32 0, !dbg !337
  %._guts._object._countAndFlagsBits._value = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %._guts._object._countAndFlagsBits, i32 0, i32 0, !dbg !337
  store i64 %17, i64* %._guts._object._countAndFlagsBits._value, align 8, !dbg !337
  %._guts._object._object = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %._guts._object, i32 0, i32 1, !dbg !337
  store %swift.bridge* %18, %swift.bridge** %._guts._object._object, align 8, !dbg !337
  %23 = call swiftcc %swift.bridge* @"$ss27_finalizeUninitializedArrayySayxGABnlF"(%swift.bridge* %13, %swift.type* getelementptr inbounds (%swift.full_type, %swift.full_type* @"$sypN", i32 0, i32 1)), !dbg !337
  %24 = call swiftcc { i64, %swift.bridge* } @"$ss5print_9separator10terminatoryypd_S2StFfA0_"(), !dbg !338
  %25 = extractvalue { i64, %swift.bridge* } %24, 0, !dbg !338
  %26 = extractvalue { i64, %swift.bridge* } %24, 1, !dbg !338
  %27 = call swiftcc { i64, %swift.bridge* } @"$ss5print_9separator10terminatoryypd_S2StFfA1_"(), !dbg !338
  %28 = extractvalue { i64, %swift.bridge* } %27, 0, !dbg !338
  %29 = extractvalue { i64, %swift.bridge* } %27, 1, !dbg !338
  call swiftcc void @"$ss5print_9separator10terminatoryypd_S2StF"(%swift.bridge* %23, i64 %25, %swift.bridge* %26, i64 %28, %swift.bridge* %29), !dbg !339
  call void @swift_bridgeObjectRelease(%swift.bridge* %29) #7, !dbg !340
  call void @swift_bridgeObjectRelease(%swift.bridge* %26) #7, !dbg !340
  call void @swift_bridgeObjectRelease(%swift.bridge* %23) #7, !dbg !340
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T27move_function_dbginfo_async5KlassC*)*)(%T27move_function_dbginfo_async5KlassC* %.reload14) #7, !dbg !340
  %toDestroy = load %T27move_function_dbginfo_async5KlassC*, %T27move_function_dbginfo_async5KlassC** %k, align 8, !dbg !340
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T27move_function_dbginfo_async5KlassC*)*)(%T27move_function_dbginfo_async5KlassC* %toDestroy) #7, !dbg !340
  %30 = bitcast %T27move_function_dbginfo_async5KlassC** %k to i8*, !dbg !340
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %30), !dbg !340
  call void asm sideeffect "", "r"(%T27move_function_dbginfo_async5KlassC** %k), !dbg !340
  call void asm sideeffect "", "r"(%T27move_function_dbginfo_async5KlassC** %m.debug), !dbg !340
  %31 = load %swift.context*, %swift.context** %1, align 8, !dbg !340
  %32 = bitcast %swift.context* %31 to <{ %swift.context*, void (%swift.context*)* }>*, !dbg !340
  %33 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)* }>, <{ %swift.context*, void (%swift.context*)* }>* %32, i32 0, i32 1, !dbg !340
  %34 = load void (%swift.context*)*, void (%swift.context*)** %33, align 8, !dbg !340
  %35 = load %swift.context*, %swift.context** %1, align 8, !dbg !340
  %36 = bitcast void (%swift.context*)* %34 to i8*, !dbg !340
  musttail call swifttailcc void %34(%swift.context* swiftasync %35) #7, !dbg !341
  ret void, !dbg !341
}

; Function Attrs: noinline nounwind readnone
declare swiftcc %swift.metadata_response @"$s27move_function_dbginfo_async5KlassCMa"(i64) #8

; Function Attrs: nounwind willreturn
declare %swift.refcounted* @swift_retain(%swift.refcounted* returned) #9

; Function Attrs: nounwind
declare void @swift_release(%swift.refcounted*) #7

declare swiftcc { %swift.bridge*, i8* } @"$ss27_allocateUninitializedArrayySayxG_BptBwlFyp_Tg5"(i64) #2

declare swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8*, i64, i1) #2

declare hidden swiftcc %swift.bridge* @"$ss27_finalizeUninitializedArrayySayxGABnlF"(%swift.bridge*, %swift.type*) #2

declare hidden swiftcc { i64, %swift.bridge* } @"$ss5print_9separator10terminatoryypd_S2StFfA0_"() #2

declare hidden swiftcc { i64, %swift.bridge* } @"$ss5print_9separator10terminatoryypd_S2StFfA1_"() #2

declare swiftcc void @"$ss5print_9separator10terminatoryypd_S2StF"(%swift.bridge*, i64, %swift.bridge*, i64, %swift.bridge*) #2

; Function Attrs: nounwind
declare void @swift_bridgeObjectRelease(%swift.bridge*) #7

; Function Attrs: alwaysinline
declare hidden void @coro.devirt.trigger(i8*) #10

attributes #0 = { argmemonly nofree nounwind willreturn writeonly }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { argmemonly nofree nosync nounwind willreturn }
attributes #5 = { nounwind readnone }
attributes #6 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #7 = { nounwind }
attributes #8 = { noinline nounwind readnone "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #9 = { nounwind willreturn }
attributes #10 = { alwaysinline }

!llvm.dbg.cu = !{!0}
!swift.module.flags = !{!11}
!llvm.asan.globals = !{!12, !13, !14, !15, !16, !17, !18, !19, !20, !21}
!llvm.module.flags = !{!22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36}
!llvm.linker.options = !{!37, !38, !39, !40, !41}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Swift version 5.7-dev (LLVM 8021856c74d1a44, Swift ea09951570b69b9)", isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, imports: !2)
!1 = !DIFile(filename: "move_function_dbginfo_async.swift", directory: "/Volumes/Data/work/solon/swift/test/DebugInfo")
!2 = !{!3, !5, !7, !9}
!3 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !4, file: !1)
!4 = !DIModule(scope: null, name: "move_function_dbginfo_async")
!5 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !6, file: !1)
!6 = !DIModule(scope: null, name: "Swift", includePath: "/Volumes/Data/work/solon/build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/Swift.swiftmodule/x86_64-apple-macos.swiftmodule")
!7 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !8, file: !1)
!8 = !DIModule(scope: null, name: "_Concurrency", includePath: "/Volumes/Data/work/solon/build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/_Concurrency.swiftmodule/x86_64-apple-macos.swiftmodule")
!9 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !10, file: !1)
!10 = !DIModule(scope: null, name: "SwiftOnoneSupport", includePath: "/Volumes/Data/work/solon/build/Ninja-RelWithDebInfoAssert/swift-macosx-x86_64/lib/swift/macosx/SwiftOnoneSupport.swiftmodule/x86_64-apple-macos.swiftmodule")
!11 = !{!"standard-library", i1 false}
!12 = distinct !{null, null, null, i1 false, i1 true}
!13 = distinct !{null, null, null, i1 false, i1 true}
!14 = distinct !{null, null, null, i1 false, i1 true}
!15 = distinct !{null, null, null, i1 false, i1 true}
!16 = distinct !{null, null, null, i1 false, i1 true}
!17 = distinct !{null, null, null, i1 false, i1 true}
!18 = distinct !{null, null, null, i1 false, i1 true}
!19 = distinct !{null, null, null, i1 false, i1 true}
!20 = distinct !{null, null, null, i1 false, i1 true}
!21 = distinct !{null, null, null, i1 false, i1 true}
!22 = !{i32 1, !"Objective-C Version", i32 2}
!23 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!24 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!25 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!26 = !{i32 1, !"Objective-C Class Properties", i32 64}
!27 = !{i32 7, !"Dwarf Version", i32 4}
!28 = !{i32 2, !"Debug Info Version", i32 3}
!29 = !{i32 1, !"wchar_size", i32 4}
!30 = !{i32 7, !"PIC Level", i32 2}
!31 = !{i32 7, !"uwtable", i32 1}
!32 = !{i32 7, !"frame-pointer", i32 2}
!33 = !{i32 1, !"Swift Version", i32 7}
!34 = !{i32 1, !"Swift ABI Version", i32 7}
!35 = !{i32 1, !"Swift Major Version", i8 5}
!36 = !{i32 1, !"Swift Minor Version", i8 7}
!37 = !{!"-lswiftSwiftOnoneSupport"}
!38 = !{!"-lswiftCore"}
!39 = !{!"-lswift_Concurrency"}
!40 = !{!"-lobjc"}
!41 = !{!"-lswiftCompatibilityConcurrency"}
!42 = distinct !DISubprogram(name: "letSimpleTest", linkageName: "$s27move_function_dbginfo_async13letSimpleTestyyxnYalF", scope: !4, file: !1, line: 78, type: !43, scopeLine: 78, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !48)
!43 = !DISubroutineType(types: !44)
!44 = !{!45, !47}
!45 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sytD", file: !1, elements: !46, runtimeLang: DW_LANG_Swift, identifier: "$sytD")
!46 = !{}
!47 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sxD", file: !1, runtimeLang: DW_LANG_Swift, identifier: "$sxD")
!48 = !{!49, !54}
!49 = !DILocalVariable(name: "$\CF\84_0_0", scope: !42, file: !1, type: !50, flags: DIFlagArtificial)
!50 = !DIDerivedType(tag: DW_TAG_typedef, name: "T", scope: !52, file: !51, baseType: !53)
!51 = !DIFile(filename: "<compiler-generated>", directory: "")
!52 = !DIModule(scope: null, name: "Builtin")
!53 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "$sBpD", baseType: null, size: 64)
!54 = !DILocalVariable(name: "msg", arg: 1, scope: !42, file: !1, line: 78, type: !55)
!55 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !47)
!56 = !DILocation(line: 0, scope: !42)
!57 = !DILocation(line: 0, scope: !58)
!58 = !DILexicalBlockFile(scope: !59, file: !51, discriminator: 0)
!59 = distinct !DILexicalBlock(scope: !42, file: !1, line: 78, column: 54)
!60 = !{i64 96}
!61 = !DILocation(line: 0, scope: !62)
!62 = !DILexicalBlockFile(scope: !42, file: !51, discriminator: 0)
!63 = !DILocation(line: 78, column: 30, scope: !42)
!64 = !DILocation(line: 79, column: 11, scope: !59)
!65 = !DILocation(line: 0, scope: !66, inlinedAt: !68)
!66 = distinct !DISubprogram(linkageName: "$s27move_function_dbginfo_async13letSimpleTestyyxnYalF", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!67 = !DISubroutineType(types: null)
!68 = distinct !DILocation(line: 79, column: 11, scope: !59)
!69 = distinct !DISubprogram(name: "letSimpleTest", linkageName: "$s27move_function_dbginfo_async13letSimpleTestyyxnYalFTQ0_", scope: !4, file: !1, line: 78, type: !43, scopeLine: 79, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !70)
!70 = !{!71, !72}
!71 = !DILocalVariable(name: "$\CF\84_0_0", scope: !69, file: !1, type: !50, flags: DIFlagArtificial)
!72 = !DILocalVariable(name: "msg", arg: 1, scope: !69, file: !1, line: 78, type: !55)
!73 = !DILocation(line: 0, scope: !69)
!74 = !DILocation(line: 0, scope: !75, inlinedAt: !76)
!75 = distinct !DISubprogram(linkageName: "__swift_async_resume_project_context", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
!76 = distinct !DILocation(line: 79, column: 11, scope: !77)
!77 = distinct !DILexicalBlock(scope: !69, file: !1, line: 78, column: 54)
!78 = !DILocation(line: 79, column: 11, scope: !77)
!79 = !DILocation(line: 0, scope: !75, inlinedAt: !80)
!80 = distinct !DILocation(line: 79, column: 11, scope: !77)
!81 = !DILocation(line: 0, scope: !82)
!82 = !DILexicalBlockFile(scope: !69, file: !51, discriminator: 0)
!83 = !DILocation(line: 78, column: 30, scope: !69)
!84 = !DILocation(line: 0, scope: !77)
!85 = !DILocation(line: 0, scope: !86, inlinedAt: !87)
!86 = distinct !DISubprogram(linkageName: "__swift_suspend_point", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!87 = distinct !DILocation(line: 0, scope: !77)
!88 = distinct !DISubprogram(name: "letSimpleTest", linkageName: "$s27move_function_dbginfo_async13letSimpleTestyyxnYalFTY1_", scope: !4, file: !1, line: 78, type: !43, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !89)
!89 = !{!90, !91}
!90 = !DILocalVariable(name: "$\CF\84_0_0", scope: !88, file: !1, type: !50, flags: DIFlagArtificial)
!91 = !DILocalVariable(name: "msg", arg: 1, scope: !88, file: !1, line: 78, type: !55)
!92 = !DILocation(line: 0, scope: !88)
!93 = !DILocation(line: 0, scope: !94)
!94 = !DILexicalBlockFile(scope: !95, file: !51, discriminator: 0)
!95 = distinct !DILexicalBlock(scope: !88, file: !1, line: 78, column: 54)
!96 = !DILocation(line: 0, scope: !95)
!97 = !DILocation(line: 0, scope: !98)
!98 = !DILexicalBlockFile(scope: !88, file: !51, discriminator: 0)
!99 = !DILocation(line: 78, column: 30, scope: !88)
!100 = !DILocation(line: 80, column: 15, scope: !95)
!101 = !DILocation(line: 80, column: 9, scope: !95)
!102 = !DILocation(line: 80, column: 5, scope: !95)
!103 = !DILocation(line: 81, column: 1, scope: !95)
!104 = !DILocation(line: 0, scope: !105, inlinedAt: !106)
!105 = distinct !DISubprogram(linkageName: "$s27move_function_dbginfo_async13letSimpleTestyyxnYalF", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!106 = distinct !DILocation(line: 81, column: 1, scope: !95)
!107 = distinct !DISubprogram(name: "varSimpleTest", linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF", scope: !4, file: !1, line: 175, type: !108, scopeLine: 175, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !110)
!108 = !DISubroutineType(types: !109)
!109 = !{!45, !47, !47}
!110 = !{!111, !112, !113}
!111 = !DILocalVariable(name: "$\CF\84_0_0", scope: !107, file: !1, type: !50, flags: DIFlagArtificial)
!112 = !DILocalVariable(name: "msg", arg: 1, scope: !107, file: !1, line: 175, type: !47)
!113 = !DILocalVariable(name: "msg2", arg: 2, scope: !107, file: !1, line: 175, type: !55)
!114 = !DILocation(line: 175, column: 46, scope: !107)
!115 = !DILocation(line: 0, scope: !107)
!116 = !DILocation(line: 0, scope: !117)
!117 = !DILexicalBlockFile(scope: !118, file: !51, discriminator: 0)
!118 = distinct !DILexicalBlock(scope: !107, file: !1, line: 175, column: 63)
!119 = !DILocation(line: 0, scope: !120)
!120 = !DILexicalBlockFile(scope: !107, file: !51, discriminator: 0)
!121 = !DILocation(line: 175, column: 30, scope: !107)
!122 = !DILocation(line: 176, column: 11, scope: !118)
!123 = !DILocation(line: 0, scope: !124, inlinedAt: !125)
!124 = distinct !DISubprogram(linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!125 = distinct !DILocation(line: 176, column: 11, scope: !118)
!126 = distinct !DISubprogram(name: "varSimpleTest", linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ0_", scope: !4, file: !1, line: 175, type: !108, scopeLine: 176, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !127)
!127 = !{!128, !129, !130}
!128 = !DILocalVariable(name: "$\CF\84_0_0", scope: !126, file: !1, type: !50, flags: DIFlagArtificial)
!129 = !DILocalVariable(name: "msg", arg: 1, scope: !126, file: !1, line: 175, type: !47)
!130 = !DILocalVariable(name: "msg2", arg: 2, scope: !126, file: !1, line: 175, type: !55)
!131 = !DILocation(line: 175, column: 46, scope: !126)
!132 = !DILocation(line: 0, scope: !126)
!133 = !DILocation(line: 0, scope: !75, inlinedAt: !134)
!134 = distinct !DILocation(line: 176, column: 11, scope: !135)
!135 = distinct !DILexicalBlock(scope: !126, file: !1, line: 175, column: 63)
!136 = !DILocation(line: 176, column: 11, scope: !135)
!137 = !DILocation(line: 0, scope: !75, inlinedAt: !138)
!138 = distinct !DILocation(line: 176, column: 11, scope: !135)
!139 = !DILocation(line: 0, scope: !140)
!140 = !DILexicalBlockFile(scope: !126, file: !51, discriminator: 0)
!141 = !DILocation(line: 175, column: 30, scope: !126)
!142 = !DILocation(line: 0, scope: !135)
!143 = !DILocation(line: 0, scope: !86, inlinedAt: !144)
!144 = distinct !DILocation(line: 0, scope: !135)
!145 = distinct !DISubprogram(name: "varSimpleTest", linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY1_", scope: !4, file: !1, line: 175, type: !108, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !146)
!146 = !{!147, !148, !149}
!147 = !DILocalVariable(name: "$\CF\84_0_0", scope: !145, file: !1, type: !50, flags: DIFlagArtificial)
!148 = !DILocalVariable(name: "msg", arg: 1, scope: !145, file: !1, line: 175, type: !47)
!149 = !DILocalVariable(name: "msg2", arg: 2, scope: !145, file: !1, line: 175, type: !55)
!150 = !DILocation(line: 175, column: 46, scope: !145)
!151 = !DILocation(line: 0, scope: !145)
!152 = !DILocation(line: 0, scope: !153)
!153 = !DILexicalBlockFile(scope: !154, file: !51, discriminator: 0)
!154 = distinct !DILexicalBlock(scope: !145, file: !1, line: 175, column: 63)
!155 = !DILocation(line: 0, scope: !154)
!156 = !DILocation(line: 0, scope: !157)
!157 = !DILexicalBlockFile(scope: !145, file: !51, discriminator: 0)
!158 = !DILocation(line: 175, column: 30, scope: !145)
!159 = !DILocation(line: 177, column: 15, scope: !154)
!160 = !DILocation(line: 177, column: 9, scope: !154)
!161 = !DILocation(line: 177, column: 5, scope: !154)
!162 = !DILocation(line: 178, column: 11, scope: !154)
!163 = !DILocation(line: 0, scope: !164, inlinedAt: !165)
!164 = distinct !DISubprogram(linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!165 = distinct !DILocation(line: 178, column: 11, scope: !154)
!166 = distinct !DISubprogram(name: "varSimpleTest", linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ2_", scope: !4, file: !1, line: 175, type: !108, scopeLine: 178, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !167)
!167 = !{!168, !169, !170}
!168 = !DILocalVariable(name: "$\CF\84_0_0", scope: !166, file: !1, type: !50, flags: DIFlagArtificial)
!169 = !DILocalVariable(name: "msg", arg: 1, scope: !166, file: !1, line: 175, type: !47)
!170 = !DILocalVariable(name: "msg2", arg: 2, scope: !166, file: !1, line: 175, type: !55)
!171 = !DILocation(line: 175, column: 46, scope: !166)
!172 = !DILocation(line: 0, scope: !166)
!173 = !DILocation(line: 0, scope: !75, inlinedAt: !174)
!174 = distinct !DILocation(line: 178, column: 11, scope: !175)
!175 = distinct !DILexicalBlock(scope: !166, file: !1, line: 175, column: 63)
!176 = !DILocation(line: 178, column: 11, scope: !175)
!177 = !DILocation(line: 0, scope: !75, inlinedAt: !178)
!178 = distinct !DILocation(line: 178, column: 11, scope: !175)
!179 = !DILocation(line: 175, column: 30, scope: !166)
!180 = !DILocation(line: 0, scope: !175)
!181 = !DILocation(line: 0, scope: !86, inlinedAt: !182)
!182 = distinct !DILocation(line: 0, scope: !175)
!183 = distinct !DISubprogram(name: "varSimpleTest", linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY3_", scope: !4, file: !1, line: 175, type: !108, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !184)
!184 = !{!185, !186, !187}
!185 = !DILocalVariable(name: "$\CF\84_0_0", scope: !183, file: !1, type: !50, flags: DIFlagArtificial)
!186 = !DILocalVariable(name: "msg", arg: 1, scope: !183, file: !1, line: 175, type: !47)
!187 = !DILocalVariable(name: "msg2", arg: 2, scope: !183, file: !1, line: 175, type: !55)
!188 = !DILocation(line: 175, column: 46, scope: !183)
!189 = !DILocation(line: 0, scope: !183)
!190 = !DILocation(line: 177, column: 9, scope: !191)
!191 = distinct !DILexicalBlock(scope: !183, file: !1, line: 175, column: 63)
!192 = !DILocation(line: 177, column: 15, scope: !191)
!193 = !DILocation(line: 0, scope: !194)
!194 = !DILexicalBlockFile(scope: !191, file: !51, discriminator: 0)
!195 = !DILocation(line: 0, scope: !191)
!196 = !DILocation(line: 175, column: 30, scope: !183)
!197 = !DILocation(line: 179, column: 11, scope: !191)
!198 = !DILocation(line: 179, column: 9, scope: !191)
!199 = !DILocation(line: 0, scope: !200)
!200 = !DILexicalBlockFile(scope: !183, file: !51, discriminator: 0)
!201 = !DILocation(line: 180, column: 22, scope: !191)
!202 = !DILocation(line: 180, column: 16, scope: !191)
!203 = !DILocation(line: 181, column: 13, scope: !191)
!204 = !DILocation(line: 182, column: 11, scope: !191)
!205 = !DILocation(line: 182, column: 9, scope: !191)
!206 = !DILocation(line: 183, column: 11, scope: !191)
!207 = !DILocation(line: 0, scope: !208, inlinedAt: !209)
!208 = distinct !DISubprogram(linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!209 = distinct !DILocation(line: 183, column: 11, scope: !191)
!210 = distinct !DISubprogram(name: "varSimpleTest", linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTQ4_", scope: !4, file: !1, line: 175, type: !108, scopeLine: 183, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !211)
!211 = !{!212, !213, !214}
!212 = !DILocalVariable(name: "$\CF\84_0_0", scope: !210, file: !1, type: !50, flags: DIFlagArtificial)
!213 = !DILocalVariable(name: "msg", arg: 1, scope: !210, file: !1, line: 175, type: !47)
!214 = !DILocalVariable(name: "msg2", arg: 2, scope: !210, file: !1, line: 175, type: !55)
!215 = !DILocation(line: 175, column: 46, scope: !210)
!216 = !DILocation(line: 0, scope: !210)
!217 = !DILocation(line: 0, scope: !75, inlinedAt: !218)
!218 = distinct !DILocation(line: 183, column: 11, scope: !219)
!219 = distinct !DILexicalBlock(scope: !210, file: !1, line: 175, column: 63)
!220 = !DILocation(line: 183, column: 11, scope: !219)
!221 = !DILocation(line: 0, scope: !75, inlinedAt: !222)
!222 = distinct !DILocation(line: 183, column: 11, scope: !219)
!223 = !DILocation(line: 0, scope: !224)
!224 = !DILexicalBlockFile(scope: !210, file: !51, discriminator: 0)
!225 = !DILocation(line: 175, column: 30, scope: !210)
!226 = !DILocation(line: 0, scope: !219)
!227 = !DILocation(line: 0, scope: !86, inlinedAt: !228)
!228 = distinct !DILocation(line: 0, scope: !219)
!229 = distinct !DISubprogram(name: "varSimpleTest", linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalFTY5_", scope: !4, file: !1, line: 175, type: !108, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !230)
!230 = !{!231, !232, !233}
!231 = !DILocalVariable(name: "$\CF\84_0_0", scope: !229, file: !1, type: !50, flags: DIFlagArtificial)
!232 = !DILocalVariable(name: "msg", arg: 1, scope: !229, file: !1, line: 175, type: !47)
!233 = !DILocalVariable(name: "msg2", arg: 2, scope: !229, file: !1, line: 175, type: !55)
!234 = !DILocation(line: 175, column: 46, scope: !229)
!235 = !DILocation(line: 0, scope: !229)
!236 = !DILocation(line: 177, column: 9, scope: !237)
!237 = distinct !DILexicalBlock(scope: !229, file: !1, line: 175, column: 63)
!238 = !DILocation(line: 0, scope: !239)
!239 = !DILexicalBlockFile(scope: !237, file: !51, discriminator: 0)
!240 = !DILocation(line: 0, scope: !237)
!241 = !DILocation(line: 0, scope: !242)
!242 = !DILexicalBlockFile(scope: !229, file: !51, discriminator: 0)
!243 = !DILocation(line: 175, column: 30, scope: !229)
!244 = !DILocation(line: 184, column: 1, scope: !237)
!245 = !DILocation(line: 0, scope: !246, inlinedAt: !247)
!246 = distinct !DISubprogram(linkageName: "$s27move_function_dbginfo_async13varSimpleTestyyxz_xtYalF", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!247 = distinct !DILocation(line: 184, column: 1, scope: !237)
!248 = distinct !DISubprogram(name: "varSimpleTestVar", linkageName: "$s27move_function_dbginfo_async16varSimpleTestVaryyYaF", scope: !4, file: !1, line: 265, type: !249, scopeLine: 265, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !251)
!249 = !DISubroutineType(types: !250)
!250 = !{!45}
!251 = !{!252, !255}
!252 = !DILocalVariable(name: "k", scope: !253, file: !1, line: 266, type: !254)
!253 = distinct !DILexicalBlock(scope: !248, file: !1, line: 265, column: 38)
!254 = !DICompositeType(tag: DW_TAG_structure_type, name: "Klass", scope: !4, file: !1, size: 64, elements: !46, runtimeLang: DW_LANG_Swift, identifier: "$s27move_function_dbginfo_async5KlassCD")
!255 = !DILocalVariable(name: "m", scope: !253, file: !1, line: 269, type: !256)
!256 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !254)
!257 = !DILocation(line: 269, column: 9, scope: !253)
!258 = !DILocation(line: 0, scope: !248)
!259 = !DILocation(line: 0, scope: !86, inlinedAt: !260)
!260 = distinct !DILocation(line: 0, scope: !248)
!261 = distinct !DISubprogram(name: "varSimpleTestVar", linkageName: "$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY0_", scope: !4, file: !1, line: 265, type: !249, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !262)
!262 = !{!263, !265}
!263 = !DILocalVariable(name: "k", scope: !264, file: !1, line: 266, type: !254)
!264 = distinct !DILexicalBlock(scope: !261, file: !1, line: 265, column: 38)
!265 = !DILocalVariable(name: "m", scope: !264, file: !1, line: 269, type: !256)
!266 = !DILocation(line: 269, column: 9, scope: !264)
!267 = !DILocation(line: 0, scope: !261)
!268 = !DILocation(line: 0, scope: !269)
!269 = !DILexicalBlockFile(scope: !264, file: !51, discriminator: 0)
!270 = !DILocation(line: 266, column: 9, scope: !264)
!271 = !DILocation(line: 266, column: 13, scope: !264)
!272 = !DILocation(line: 267, column: 7, scope: !264)
!273 = !DILocation(line: 268, column: 11, scope: !264)
!274 = !DILocation(line: 0, scope: !275, inlinedAt: !276)
!275 = distinct !DISubprogram(linkageName: "$s27move_function_dbginfo_async16varSimpleTestVaryyYaF", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!276 = distinct !DILocation(line: 268, column: 11, scope: !264)
!277 = distinct !DISubprogram(name: "varSimpleTestVar", linkageName: "$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTQ1_", scope: !4, file: !1, line: 265, type: !249, scopeLine: 268, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !278)
!278 = !{!279, !281}
!279 = !DILocalVariable(name: "k", scope: !280, file: !1, line: 266, type: !254)
!280 = distinct !DILexicalBlock(scope: !277, file: !1, line: 265, column: 38)
!281 = !DILocalVariable(name: "m", scope: !280, file: !1, line: 269, type: !256)
!282 = !DILocation(line: 269, column: 9, scope: !280)
!283 = !DILocation(line: 0, scope: !75, inlinedAt: !284)
!284 = distinct !DILocation(line: 268, column: 11, scope: !280)
!285 = !DILocation(line: 268, column: 11, scope: !280)
!286 = !DILocation(line: 0, scope: !75, inlinedAt: !287)
!287 = distinct !DILocation(line: 268, column: 11, scope: !280)
!288 = !DILocation(line: 0, scope: !289)
!289 = !DILexicalBlockFile(scope: !280, file: !51, discriminator: 0)
!290 = !DILocation(line: 266, column: 9, scope: !280)
!291 = !DILocation(line: 0, scope: !86, inlinedAt: !292)
!292 = distinct !DILocation(line: 0, scope: !289)
!293 = distinct !DISubprogram(name: "varSimpleTestVar", linkageName: "$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY2_", scope: !4, file: !1, line: 265, type: !249, scopeLine: 265, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !294)
!294 = !{!295, !297}
!295 = !DILocalVariable(name: "k", scope: !296, file: !1, line: 266, type: !254)
!296 = distinct !DILexicalBlock(scope: !293, file: !1, line: 265, column: 38)
!297 = !DILocalVariable(name: "m", scope: !296, file: !1, line: 269, type: !256)
!298 = !DILocation(line: 269, column: 9, scope: !296)
!299 = !DILocation(line: 0, scope: !300)
!300 = !DILexicalBlockFile(scope: !296, file: !51, discriminator: 0)
!301 = !DILocation(line: 266, column: 9, scope: !296)
!302 = !DILocation(line: 269, column: 13, scope: !296)
!303 = !DILocation(line: 0, scope: !296)
!304 = !DILocation(line: 270, column: 7, scope: !296)
!305 = !DILocation(line: 271, column: 11, scope: !296)
!306 = !DILocation(line: 0, scope: !307, inlinedAt: !308)
!307 = distinct !DISubprogram(linkageName: "$s27move_function_dbginfo_async16varSimpleTestVaryyYaF", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!308 = distinct !DILocation(line: 271, column: 11, scope: !296)
!309 = distinct !DISubprogram(name: "varSimpleTestVar", linkageName: "$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTQ3_", scope: !4, file: !1, line: 265, type: !249, scopeLine: 271, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !310)
!310 = !{!311, !313}
!311 = !DILocalVariable(name: "k", scope: !312, file: !1, line: 266, type: !254)
!312 = distinct !DILexicalBlock(scope: !309, file: !1, line: 265, column: 38)
!313 = !DILocalVariable(name: "m", scope: !312, file: !1, line: 269, type: !256)
!314 = !DILocation(line: 269, column: 9, scope: !312)
!315 = !DILocation(line: 0, scope: !75, inlinedAt: !316)
!316 = distinct !DILocation(line: 271, column: 11, scope: !312)
!317 = !DILocation(line: 271, column: 11, scope: !312)
!318 = !DILocation(line: 0, scope: !75, inlinedAt: !319)
!319 = distinct !DILocation(line: 271, column: 11, scope: !312)
!320 = !DILocation(line: 266, column: 9, scope: !312)
!321 = !DILocation(line: 0, scope: !322)
!322 = !DILexicalBlockFile(scope: !312, file: !51, discriminator: 0)
!323 = !DILocation(line: 0, scope: !86, inlinedAt: !324)
!324 = distinct !DILocation(line: 0, scope: !322)
!325 = distinct !DISubprogram(name: "varSimpleTestVar", linkageName: "$s27move_function_dbginfo_async16varSimpleTestVaryyYaFTY4_", scope: !4, file: !1, line: 265, type: !249, scopeLine: 265, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !326)
!326 = !{!327, !329}
!327 = !DILocalVariable(name: "k", scope: !328, file: !1, line: 266, type: !254)
!328 = distinct !DILexicalBlock(scope: !325, file: !1, line: 265, column: 38)
!329 = !DILocalVariable(name: "m", scope: !328, file: !1, line: 269, type: !256)
!330 = !DILocation(line: 269, column: 9, scope: !328)
!331 = !DILocation(line: 0, scope: !332)
!332 = !DILexicalBlockFile(scope: !328, file: !51, discriminator: 0)
!333 = !DILocation(line: 266, column: 9, scope: !328)
!334 = !DILocation(line: 272, column: 9, scope: !328)
!335 = !DILocation(line: 272, column: 7, scope: !328)
!336 = !DILocation(line: 273, column: 7, scope: !328)
!337 = !DILocation(line: 274, column: 11, scope: !328)
!338 = !DILocation(line: 274, column: 10, scope: !328)
!339 = !DILocation(line: 274, column: 5, scope: !328)
!340 = !DILocation(line: 275, column: 1, scope: !328)
!341 = !DILocation(line: 0, scope: !342, inlinedAt: !343)
!342 = distinct !DISubprogram(linkageName: "$s27move_function_dbginfo_async16varSimpleTestVaryyYaF", scope: !4, file: !51, type: !67, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !46)
!343 = distinct !DILocation(line: 275, column: 1, scope: !328)
