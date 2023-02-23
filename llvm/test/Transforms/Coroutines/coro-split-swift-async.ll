; RUN: opt < %s -O0 -S | FileCheck %s
; RUN: opt < %s -passes='default<O0>' -S | FileCheck %s
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

; CHECK: define internal swiftcc void @"$s1a1fyS2iYFTY0_"(ptr %0, ptr %1, ptr swiftasync %2)
; CHECK: entryresume.0:
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata ptr %2, metadata ![[X:[0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 64, DW_OP_plus_uconst, 24))
; CHECK: ![[X]] = !DILocalVariable(name: "x",
source_filename = "/tmp/a.ll"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "arm64e-apple-macosx11.0.0"

%swift.async_func_pointer = type <{ i32, i32 }>
%TSi = type <{ i64 }>
%swift.metadata_response = type { ptr, i64 }
%swift.task = type { %swift.refcounted, ptr, ptr, i64, ptr, ptr, i64 }
%swift.refcounted = type { ptr, i64 }

@"$s1a1fyS2iYFTu" = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1a1fyS2iYF" to i64), i64 ptrtoint (ptr @"$s1a1fyS2iYFTu" to i64)) to i32), i32 64 }>, section "__TEXT,__const", align 8

define hidden swiftcc void @"$s1a1fyS2iYF"(ptr %0, ptr %1, ptr swiftasync %2) #0 !dbg !41 {
entry:
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  store ptr %2, ptr %5, align 8
  %6 = bitcast ptr %2 to ptr
  %7 = call token @llvm.coro.id.async(i32 64, i32 16, i32 2, ptr @"$s1a1fyS2iYFTu")
  %8 = call noalias nonnull ptr @llvm.coro.begin(token %7, ptr null)
  %9 = getelementptr inbounds <{ ptr, ptr, ptr, i32, [4 x i8], ptr, %TSi, ptr, %TSi }>, ptr %6, i32 0, i32 7
  %10 = load ptr, ptr %9, align 8
  %11 = getelementptr inbounds <{ ptr, ptr, ptr, i32, [4 x i8], ptr, %TSi, ptr, %TSi }>, ptr %6, i32 0, i32 8
  %._value = getelementptr inbounds %TSi, ptr %11, i32 0, i32 0
  %12 = load i64, ptr %._value, align 8
  call void @llvm.dbg.declare(metadata i64 %12, metadata !46, metadata !DIExpression()), !dbg !48
  %13 = call swiftcc %swift.metadata_response @"$s12_Concurrency9MainActorCMa"(i64 0) #4, !dbg !49
  %14 = extractvalue %swift.metadata_response %13, 0, !dbg !49
  %15 = call swiftcc ptr @"$s12_Concurrency9MainActorC6sharedAC5_ImplCvgZ"(ptr swiftself %14), !dbg !49
  %16 = call ptr @llvm.coro.async.resume(), !dbg !49
  %17 = load ptr, ptr %3, align 8, !dbg !49
  %18 = load ptr, ptr %4, align 8, !dbg !49
  %19 = load ptr, ptr %3, align 8, !dbg !49
  %20 = load ptr, ptr %4, align 8, !dbg !49
  %21 = load ptr, ptr %5, align 8, !dbg !49
  %22 = load ptr, ptr %4, align 8, !dbg !49
  %23 = bitcast ptr %15 to ptr, !dbg !49
  %24 = load ptr, ptr %4, align 8, !dbg !49
  %25 = load ptr, ptr %5, align 8, !dbg !49
  %26 = call { ptr, ptr, ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async(i32 2, ptr %16, ptr @__swift_async_resume_get_context, ptr @__swift_suspend_point, ptr %16, ptr %23, ptr %17, ptr %24, ptr %25), !dbg !49
  %27 = extractvalue { ptr, ptr, ptr } %26, 0, !dbg !49
  %28 = bitcast ptr %27 to ptr, !dbg !49
  store ptr %28, ptr %3, align 8, !dbg !49
  %29 = extractvalue { ptr, ptr, ptr } %26, 1, !dbg !49
  %30 = bitcast ptr %29 to ptr, !dbg !49
  store ptr %30, ptr %4, align 8, !dbg !49
  %31 = extractvalue { ptr, ptr, ptr } %26, 2, !dbg !49
  %32 = call ptr @__swift_async_resume_get_context(ptr %31), !dbg !49
  %33 = bitcast ptr %32 to ptr, !dbg !49
  store ptr %33, ptr %5, align 8, !dbg !49
  call void @swift_release(ptr %15) #1, !dbg !50
  %34 = load ptr, ptr %5, align 8, !dbg !50
  %35 = bitcast ptr %34 to ptr, !dbg !50
  %36 = getelementptr inbounds <{ ptr, ptr, ptr, i32, [4 x i8], ptr, %TSi, ptr, %TSi }>, ptr %35, i32 0, i32 6, !dbg !50
  %._value1 = getelementptr inbounds %TSi, ptr %36, i32 0, i32 0, !dbg !50
  store i64 %12, ptr %._value1, align 8, !dbg !50
  %37 = load ptr, ptr %5, align 8, !dbg !50
  %38 = bitcast ptr %37 to ptr, !dbg !50
  %39 = getelementptr inbounds <{ ptr, ptr, ptr, i32, [4 x i8], ptr, %TSi, ptr, %TSi }>, ptr %38, i32 0, i32 1, !dbg !50
  %40 = load ptr, ptr %39, align 8, !dbg !50
  %41 = load ptr, ptr %3, align 8, !dbg !50
  %42 = load ptr, ptr %4, align 8, !dbg !50
  %43 = load ptr, ptr %5, align 8, !dbg !50
  tail call swiftcc void %40(ptr %41, ptr %42, ptr swiftasync %43), !dbg !50
  br label %coro.end, !dbg !50

coro.end:                                         ; preds = %entry
  %44 = call i1 @llvm.coro.end(ptr %8, i1 false) #5, !dbg !50
  unreachable, !dbg !50
}

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, ptr) #1

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare swiftcc %swift.metadata_response @"$s12_Concurrency9MainActorCMa"(i64) #0

declare swiftcc ptr @"$s12_Concurrency9MainActorC6sharedAC5_ImplCvgZ"(ptr swiftself) #0

; Function Attrs: nounwind
declare ptr @llvm.coro.async.resume() #1

; Function Attrs: nounwind
define linkonce_odr hidden ptr @__swift_async_resume_get_context(ptr %0) #4 {
entry:
  ret ptr %0
}

; Function Attrs: nounwind
define internal void @__swift_suspend_point(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) #1 {
entry:
  %5 = getelementptr inbounds %swift.task, ptr %2, i32 0, i32 4
  store ptr %0, ptr %5, align 8
  %6 = getelementptr inbounds %swift.task, ptr %2, i32 0, i32 5
  store ptr %4, ptr %6, align 8
  tail call swiftcc void @swift_task_switch(ptr %2, ptr %3, ptr %1) #1
  ret void
}

; Function Attrs: nounwind
declare extern_weak swiftcc void @swift_task_switch(ptr %0, ptr %1, ptr %2) #1

; Function Attrs: nounwind
declare { ptr, ptr, ptr } @llvm.coro.suspend.async(i32, ptr, ptr, ...) #1

; Function Attrs: nounwind
declare void @swift_release(ptr) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.end(ptr, i1) #1

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
