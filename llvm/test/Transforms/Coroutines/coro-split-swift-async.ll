; RUN: opt < %s -O0 -enable-coroutines -S | FileCheck %s
; RUN: opt < %s -passes='default<O0>' -enable-coroutines -S | FileCheck %s
;
; llvm-extracted from:
;
; @MainActor func f(_ x : Int) async -> Int {
;   return x
; }
; @main struct Main {
;   static func main() async {
;     let x = await f(23);
;   }
; }

; Test that swiftasync parameters are described by a direct
; dbg.declare and not stashed into an alloca. They will get lowered to
; an entry value in the backend.

; CHECK: define internal swiftcc void @"$s1a1fyS2iYFTY0_"(i8* %0, i8* %1, i8* swiftasync %2)
; CHECK: entryresume.0:
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata i8* %2, metadata ![[X:[0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 64, DW_OP_plus_uconst, 24))
; CHECK: ![[X]] = !DILocalVariable(name: "x",
source_filename = "/tmp/a.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "arm64e-apple-macosx11.0.0"

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.task = type { %swift.refcounted, i8*, i8*, i64, i8*, %swift.context*, i64 }
%swift.refcounted = type { %swift.type*, i64 }
%swift.type = type { i64 }
%swift.executor = type {}
%swift.context = type {}
%swift.error = type opaque
%TSi = type <{ i64 }>
%swift.metadata_response = type { %swift.type*, i64 }
%T12_Concurrency9MainActorC5_ImplC = type opaque

@"$s1a1fyS2iYFTu" = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.task*, %swift.executor*, %swift.context*)* @"$s1a1fyS2iYF" to i64), i64 ptrtoint (%swift.async_func_pointer* @"$s1a1fyS2iYFTu" to i64)) to i32), i32 64 }>, section "__TEXT,__const", align 8

define hidden swiftcc void @"$s1a1fyS2iYF"(%swift.task* %0, %swift.executor* %1, %swift.context* swiftasync %2) #0 !dbg !41 {
entry:
  %3 = alloca %swift.task*, align 8
  %4 = alloca %swift.executor*, align 8
  %5 = alloca %swift.context*, align 8
  store %swift.task* %0, %swift.task** %3, align 8
  store %swift.executor* %1, %swift.executor** %4, align 8
  store %swift.context* %2, %swift.context** %5, align 8
  %6 = bitcast %swift.context* %2 to <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>*
  %7 = call token @llvm.coro.id.async(i32 64, i32 16, i32 2, i8* bitcast (%swift.async_func_pointer* @"$s1a1fyS2iYFTu" to i8*))
  %8 = call noalias nonnull i8* @llvm.coro.begin(token %7, i8* null)
  %9 = getelementptr inbounds <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>, <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>* %6, i32 0, i32 7
  %10 = load %swift.refcounted*, %swift.refcounted** %9, align 8
  %11 = getelementptr inbounds <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>, <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>* %6, i32 0, i32 8
  %._value = getelementptr inbounds %TSi, %TSi* %11, i32 0, i32 0
  %12 = load i64, i64* %._value, align 8
  call void @llvm.dbg.declare(metadata i64 %12, metadata !46, metadata !DIExpression()), !dbg !48
  %13 = call swiftcc %swift.metadata_response @"$s12_Concurrency9MainActorCMa"(i64 0) #4, !dbg !49
  %14 = extractvalue %swift.metadata_response %13, 0, !dbg !49
  %15 = call swiftcc %T12_Concurrency9MainActorC5_ImplC* @"$s12_Concurrency9MainActorC6sharedAC5_ImplCvgZ"(%swift.type* swiftself %14), !dbg !49
  %16 = call i8* @llvm.coro.async.resume(), !dbg !49
  %17 = load %swift.task*, %swift.task** %3, align 8, !dbg !49
  %18 = load %swift.executor*, %swift.executor** %4, align 8, !dbg !49
  %19 = load %swift.task*, %swift.task** %3, align 8, !dbg !49
  %20 = load %swift.executor*, %swift.executor** %4, align 8, !dbg !49
  %21 = load %swift.context*, %swift.context** %5, align 8, !dbg !49
  %22 = load %swift.executor*, %swift.executor** %4, align 8, !dbg !49
  %23 = bitcast %T12_Concurrency9MainActorC5_ImplC* %15 to %swift.executor*, !dbg !49
  %24 = load %swift.executor*, %swift.executor** %4, align 8, !dbg !49
  %25 = load %swift.context*, %swift.context** %5, align 8, !dbg !49
  %26 = call { i8*, i8*, i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async(i32 2, i8* %16, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.executor*, %swift.task*, %swift.executor*, %swift.context*)* @__swift_suspend_point to i8*), i8* %16, %swift.executor* %23, %swift.task* %17, %swift.executor* %24, %swift.context* %25), !dbg !49
  %27 = extractvalue { i8*, i8*, i8* } %26, 0, !dbg !49
  %28 = bitcast i8* %27 to %swift.task*, !dbg !49
  store %swift.task* %28, %swift.task** %3, align 8, !dbg !49
  %29 = extractvalue { i8*, i8*, i8* } %26, 1, !dbg !49
  %30 = bitcast i8* %29 to %swift.executor*, !dbg !49
  store %swift.executor* %30, %swift.executor** %4, align 8, !dbg !49
  %31 = extractvalue { i8*, i8*, i8* } %26, 2, !dbg !49
  %32 = call i8* @__swift_async_resume_get_context(i8* %31), !dbg !49
  %33 = bitcast i8* %32 to %swift.context*, !dbg !49
  store %swift.context* %33, %swift.context** %5, align 8, !dbg !49
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T12_Concurrency9MainActorC5_ImplC*)*)(%T12_Concurrency9MainActorC5_ImplC* %15) #1, !dbg !50
  %34 = load %swift.context*, %swift.context** %5, align 8, !dbg !50
  %35 = bitcast %swift.context* %34 to <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>*, !dbg !50
  %36 = getelementptr inbounds <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>, <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>* %35, i32 0, i32 6, !dbg !50
  %._value1 = getelementptr inbounds %TSi, %TSi* %36, i32 0, i32 0, !dbg !50
  store i64 %12, i64* %._value1, align 8, !dbg !50
  %37 = load %swift.context*, %swift.context** %5, align 8, !dbg !50
  %38 = bitcast %swift.context* %37 to <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>*, !dbg !50
  %39 = getelementptr inbounds <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>, <{ %swift.context*, void (%swift.task*, %swift.executor*, %swift.context*)*, %swift.executor*, i32, [4 x i8], %swift.error*, %TSi, %swift.refcounted*, %TSi }>* %38, i32 0, i32 1, !dbg !50
  %40 = load void (%swift.task*, %swift.executor*, %swift.context*)*, void (%swift.task*, %swift.executor*, %swift.context*)** %39, align 8, !dbg !50
  %41 = load %swift.task*, %swift.task** %3, align 8, !dbg !50
  %42 = load %swift.executor*, %swift.executor** %4, align 8, !dbg !50
  %43 = load %swift.context*, %swift.context** %5, align 8, !dbg !50
  tail call swiftcc void %40(%swift.task* %41, %swift.executor* %42, %swift.context* swiftasync %43), !dbg !50
  br label %coro.end, !dbg !50

coro.end:                                         ; preds = %entry
  %44 = call i1 @llvm.coro.end(i8* %8, i1 false) #5, !dbg !50
  unreachable, !dbg !50
}

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, i8*) #1

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare swiftcc %swift.metadata_response @"$s12_Concurrency9MainActorCMa"(i64) #0

declare swiftcc %T12_Concurrency9MainActorC5_ImplC* @"$s12_Concurrency9MainActorC6sharedAC5_ImplCvgZ"(%swift.type* swiftself) #0

; Function Attrs: nounwind
declare i8* @llvm.coro.async.resume() #1

; Function Attrs: nounwind
define linkonce_odr hidden i8* @__swift_async_resume_get_context(i8* %0) #4 {
entry:
  ret i8* %0
}

; Function Attrs: nounwind
define internal void @__swift_suspend_point(i8* %0, %swift.executor* %1, %swift.task* %2, %swift.executor* %3, %swift.context* %4) #1 {
entry:
  %5 = getelementptr inbounds %swift.task, %swift.task* %2, i32 0, i32 4
  store i8* %0, i8** %5, align 8
  %6 = getelementptr inbounds %swift.task, %swift.task* %2, i32 0, i32 5
  store %swift.context* %4, %swift.context** %6, align 8
  tail call swiftcc void @swift_task_switch(%swift.task* %2, %swift.executor* %3, %swift.executor* %1) #1
  ret void
}

; Function Attrs: nounwind
declare extern_weak swiftcc void @swift_task_switch(%swift.task* %0, %swift.executor* %1, %swift.executor* %2) #1

; Function Attrs: nounwind
declare { i8*, i8*, i8* } @llvm.coro.suspend.async(i32, i8*, i8*, ...) #1

; Function Attrs: nounwind
declare void @swift_release(%swift.refcounted*) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.end(i8*, i1) #1

attributes #0 = { "frame-pointer"="all" }
attributes #1 = { nounwind }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { nounwind "frame-pointer"="all" }
attributes #4 = { nounwind readnone }
attributes #5 = { noduplicate }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13}
!llvm.dbg.cu = !{!14}
!swift.module.flags = !{!26}
!llvm.linker.options = !{!37, !38, !39, !40}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 3]}
!1 = !{i32 1, !"Objective-C Version", i32 2}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!4 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!5 = !{i32 1, !"Objective-C Class Properties", i32 64}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"PIC Level", i32 2}
!10 = !{i32 1, !"Swift Version", i32 7}
!11 = !{i32 1, !"Swift ABI Version", i32 7}
!12 = !{i32 1, !"Swift Major Version", i8 5}
!13 = !{i32 1, !"Swift Minor Version", i8 4}
!14 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !15, producer: "Swift version 5.4-dev effective-4.1.50 (LLVM 62cfff1f1fa7547, Swift f0d361a488e72b6)", isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, enums: !16, imports: !17, sysroot: "/", sdk: "MacOSX.sdk")
!15 = !DIFile(filename: "/tmp/ex1.swift", directory: "/")
!16 = !{}
!17 = !{!18, !20, !22, !24}
!18 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !15, entity: !19, file: !15)
!19 = !DIModule(scope: null, name: "a", includePath: "/tmp")
!20 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !15, entity: !21, file: !15)
!21 = !DIModule(scope: null, name: "Swift", includePath: "/")
!22 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !15, entity: !23, file: !15)
!23 = !DIModule(scope: null, name: "_Concurrency", includePath: "/")
!24 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !15, entity: !25, file: !15)
!25 = !DIModule(scope: null, name: "SwiftOnoneSupport", includePath: "/")
!26 = !{!"standard-library", i1 false}
!37 = !{!"-lswiftSwiftOnoneSupport"}
!38 = !{!"-lswiftCore"}
!39 = !{!"-lswift_Concurrency"}
!40 = !{!"-lobjc"}
!41 = distinct !DISubprogram(name: "f", linkageName: "$s1a1fyS2iYF", scope: !19, file: !15, line: 1, type: !42, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !14, retainedNodes: !16)
!42 = !DISubroutineType(types: !43)
!43 = !{!44, !44}
!44 = !DICompositeType(tag: DW_TAG_structure_type, name: "Int", scope: !21, file: !45, size: 64, elements: !16, runtimeLang: DW_LANG_Swift, identifier: "$sSiD")
!45 = !DIFile(filename: "Swift.swiftmodule/x86_64-apple-macos.swiftmodule", directory: "/")
!46 = !DILocalVariable(name: "x", arg: 1, scope: !41, file: !15, line: 1, type: !47)
!47 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !44)
!48 = !DILocation(line: 1, column: 19, scope: !41)
!49 = !DILocation(line: 1, column: 17, scope: !41)
!50 = !DILocation(line: 2, column: 3, scope: !51)
!51 = distinct !DILexicalBlock(scope: !41, file: !15, line: 1, column: 43)
