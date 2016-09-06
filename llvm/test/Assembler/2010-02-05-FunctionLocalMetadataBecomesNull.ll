; RUN: opt -S -O3 < %s | FileCheck %s
; RUN: verify-uselistorder %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.2"

%struct.anon = type { i32, i32 }
%struct.test = type { i64, %struct.anon, %struct.test* }

@TestArrayPtr = global %struct.test* getelementptr inbounds ([10 x %struct.test], [10 x %struct.test]* @TestArray, i64 0, i64 3) ; <%struct.test**> [#uses=1]
@TestArray = common global [10 x %struct.test] zeroinitializer, align 32 ; <[10 x %struct.test]*> [#uses=2]

define i32 @main() nounwind readonly !dbg !1 {
  %diff1 = alloca i64                             ; <i64*> [#uses=2]
; CHECK: call void @llvm.dbg.value(metadata i64 72,
  call void @llvm.dbg.declare(metadata i64* %diff1, metadata !0, metadata !DIExpression()), !dbg !DILocation(scope: !1)
  store i64 72, i64* %diff1, align 8
  %v1 = load %struct.test*, %struct.test** @TestArrayPtr, align 8 ; <%struct.test*> [#uses=1]
  %v2 = ptrtoint %struct.test* %v1 to i64 ; <i64> [#uses=1]
  %v3 = sub i64 %v2, ptrtoint ([10 x %struct.test]* @TestArray to i64) ; <i64> [#uses=1]
  store i64 %v3, i64* %diff1, align 8
  ret i32 4, !dbg !DILocation(scope: !1)
}

!llvm.dbg.cu = !{!6}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!7 = !{!1}
!6 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 131941)", isOptimized: true, emissionKind: FullDebug, file: !8, enums: !9, retainedTypes: !9)
!0 = !DILocalVariable(name: "c", line: 2, scope: !1, file: !2, type: !5)
!1 = distinct !DISubprogram(name: "main", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !6, scopeLine: 1, file: !8, scope: !2, type: !3)
!2 = !DIFile(filename: "/d/j/debug-test.c", directory: "/Volumes/Data/b")
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DIFile(filename: "/d/j/debug-test.c", directory: "/Volumes/Data/b")
!9 = !{}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"Debug Info Version", i32 3}
