; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -S | FileCheck %s

; We want to check that updating the declaration when updating the linkage name of a DISubporgram with a declaration.

; Original source code:
; public enum Foo {
;   public func bar() async {
;     await f()
;   }
; }
; public func f() async {}


target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx12.0.0"

%swift.async_func_pointer = type <{ i32, i32 }>

@"$s3foo3FooO3baryyYaFTu" = global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s3foo3FooO3baryyYaF" to i64), i64 ptrtoint (ptr @"$s3foo3FooO3baryyYaFTu" to i64)) to i32), i32 16 }>, align 8
@"$s3foo1fyyYaFTu" = global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s3foo1fyyYaF" to i64), i64 ptrtoint (ptr @"$s3foo1fyyYaFTu" to i64)) to i32), i32 16 }>, align 8

define swifttailcc void @"$s3foo3FooO3baryyYaF"(ptr swiftasync %0) !dbg !5 {
entry:
  %1 = alloca ptr, align 8
  %2 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @"$s3foo3FooO3baryyYaFTu")
  %3 = call ptr @llvm.coro.begin(token %2, ptr null)
  store ptr %0, ptr %1, align 8
  %4 = load i32, ptr getelementptr inbounds (%swift.async_func_pointer, ptr @"$s3foo1fyyYaFTu", i32 0, i32 1), align 8, !dbg !10
  %5 = zext i32 %4 to i64, !dbg !10
  %6 = call swiftcc ptr @swift_task_alloc(i64 %5), !dbg !10
  %7 = load ptr, ptr %1, align 8, !dbg !10
  %8 = getelementptr inbounds <{ ptr, ptr }>, ptr %6, i32 0, i32 0, !dbg !10
  store ptr %7, ptr %8, align 8, !dbg !10
  %9 = call ptr @llvm.coro.async.resume(), !dbg !10
  %10 = getelementptr inbounds <{ ptr, ptr }>, ptr %6, i32 0, i32 1, !dbg !10
  store ptr %9, ptr %10, align 8, !dbg !10
  %11 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %9, ptr @__swift_async_resume_project_context, ptr @"$s3foo3FooO3baryyYaF.0", ptr @"$s3foo1fyyYaF", ptr %6), !dbg !10
  %12 = extractvalue { ptr } %11, 0, !dbg !10
  %13 = call ptr @__swift_async_resume_project_context(ptr %12), !dbg !10
  store ptr %13, ptr %1, align 8, !dbg !10
  call swiftcc void @swift_task_dealloc(ptr %6), !dbg !10
  %14 = load ptr, ptr %1, align 8, !dbg !11
  %15 = getelementptr inbounds <{ ptr, ptr }>, ptr %14, i32 0, i32 1, !dbg !11
  %16 = load ptr, ptr %15, align 8, !dbg !11
  %17 = load ptr, ptr %1, align 8, !dbg !11
  call void (ptr, i1, ...) @llvm.coro.end.async(ptr %3, i1 false, ptr @"$s3foo3FooO3baryyYaF.0.1", ptr %16, ptr %17), !dbg !11
  unreachable, !dbg !11
}

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, ptr) #0

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #0

declare swifttailcc void @"$s3foo1fyyYaF"(ptr swiftasync)

declare swiftcc ptr @swift_task_alloc(i64)

; Function Attrs: nomerge nounwind
declare ptr @llvm.coro.async.resume() #1

define linkonce_odr hidden ptr @__swift_async_resume_project_context(ptr %0) !dbg !12 {
entry:
  %1 = load ptr, ptr %0, align 8, !dbg !14
  %2 = call ptr @llvm.swift.async.context.addr(), !dbg !14
  store ptr %1, ptr %2, align 8, !dbg !14
  ret ptr %1, !dbg !14
}

; Function Attrs: nounwind
declare ptr @llvm.swift.async.context.addr() #0

define internal swifttailcc void @"$s3foo3FooO3baryyYaF.0"(ptr %0, ptr %1) !dbg !15 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1), !dbg !16
  ret void, !dbg !16
}

; Function Attrs: nomerge nounwind
declare { ptr } @llvm.coro.suspend.async.sl_p0s(i32, ptr, ptr, ...) #1

declare swiftcc void @swift_task_dealloc(ptr)

define internal swifttailcc void @"$s3foo3FooO3baryyYaF.0.1"(ptr %0, ptr %1) !dbg !17 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1), !dbg !18
  ret void, !dbg !18
}

; Function Attrs: nounwind
declare void @llvm.coro.end.async(ptr, i1, ...) #0

attributes #0 = { nounwind }
attributes #1 = { nomerge nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Apple Swift version 5.9-dev (LLVM 1c4b88beb62789b, Swift b00d1520f89bb7d)", isOptimized: false, runtimeVersion: 5, emissionKind: LineTablesOnly, imports: !2)
!1 = !DIFile(filename: "foo.swift", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
; CHECK-DAG: ![[DECL:[0-9]+]] = !DISubprogram({{.*}}, linkageName: "$s3foo3FooO3baryyYaF"
; CHECK-DAG: ![[DECL_Q0:[0-9]+]] = !DISubprogram({{.*}}, linkageName: "$s3foo3FooO3baryyYaFTQ0_"
; CHECK-DAG: distinct !DISubprogram({{.*}}, linkageName: "$s3foo3FooO3baryyYaF"{{.*}}, declaration: ![[DECL]]
; CHECK-DAG: distinct !DISubprogram({{.*}}, linkageName: "$s3foo3FooO3baryyYaFTQ0_"{{.*}}, declaration: ![[DECL_Q0]]
!5 = distinct !DISubprogram(name: "bar", linkageName: "$s3foo3FooO3baryyYaF", scope: !6, file: !1, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, declaration: !9, retainedNodes: !2)
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "$s3foo3FooOD", scope: !7, flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift)
!7 = !DIModule(scope: null, name: "foo")
!8 = !DISubroutineType(types: null)
!9 = !DISubprogram(name: "bar", linkageName: "$s3foo3FooO3baryyYaF", scope: !6, file: !1, line: 2, type: !8, scopeLine: 2, spFlags: 0)
!10 = !DILocation(line: 3, column: 11, scope: !5)
!11 = !DILocation(line: 4, column: 3, scope: !5)
!12 = distinct !DISubprogram(linkageName: "__swift_async_resume_project_context", scope: !7, file: !13, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!13 = !DIFile(filename: "<compiler-generated>", directory: "")
!14 = !DILocation(line: 0, scope: !12)
!15 = distinct !DISubprogram(linkageName: "$s3foo3FooO3baryyYaF", scope: !7, file: !13, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 0, scope: !15)
!17 = distinct !DISubprogram(linkageName: "$s3foo3FooO3baryyYaF", scope: !7, file: !13, type: !8, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 0, scope: !17)
