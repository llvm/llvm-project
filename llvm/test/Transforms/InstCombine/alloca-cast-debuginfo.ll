; RUN: opt < %s -S -instcombine -instcombine-lower-dbg-declare=0 | FileCheck %s

; In this example, instcombine wants to turn "local" into an i64, since that's
; how it is stored. It should keep the debug info referring to the alloca when
; it does the replacement.

; C source:
; struct Foo {
;   int x, y;
; };
; void escape(const void*);
; void f(struct Foo *p) {
;   struct Foo local;
;   *(__int64 *)&local = *(__int64 *)p;
;   escape(&local);
; }

; ModuleID = '<stdin>'
source_filename = "t.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.25508"

%struct.Foo = type { i32, i32 }

define void @f(%struct.Foo* %p) !dbg !11 {
entry:
  %local = alloca %struct.Foo, align 4
  %0 = bitcast %struct.Foo* %local to i8*, !dbg !24
  call void @llvm.dbg.declare(metadata %struct.Foo* %local, metadata !22, metadata !DIExpression()), !dbg !25
  %1 = bitcast %struct.Foo* %p to i64*, !dbg !26
  %2 = load i64, i64* %1, align 8, !dbg !26, !tbaa !27
  %3 = bitcast %struct.Foo* %local to i64*, !dbg !31
  store i64 %2, i64* %3, align 4, !dbg !32, !tbaa !27
  %4 = bitcast %struct.Foo* %local to i8*, !dbg !33
  call void @escape(i8* %4), !dbg !34
  %5 = bitcast %struct.Foo* %local to i8*, !dbg !35
  ret void, !dbg !35
}

; CHECK-LABEL: define void @f(%struct.Foo* %p)
; CHECK: %local = alloca i64, align 8
; CHECK: call void @llvm.dbg.declare(metadata i64* %local, metadata !22, metadata !DIExpression())

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @escape(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "d7473625866433067a75fd7d03d2abf7")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!6 = !{i32 2, !"CodeView", i32 1}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 2}
!9 = !{i32 7, !"PIC Level", i32 2}
!10 = !{!"clang version 6.0.0 "}
!11 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 5, type: !12, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !20)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, size: 64, elements: !16)
!16 = !{!17, !19}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !15, file: !1, line: 2, baseType: !18, size: 32)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !15, file: !1, line: 2, baseType: !18, size: 32, offset: 32)
!20 = !{!21, !22}
!21 = !DILocalVariable(name: "p", arg: 1, scope: !11, file: !1, line: 5, type: !14)
!22 = !DILocalVariable(name: "local", scope: !11, file: !1, line: 6, type: !15)
!23 = !DILocation(line: 5, column: 20, scope: !11)
!24 = !DILocation(line: 6, column: 3, scope: !11)
!25 = !DILocation(line: 6, column: 14, scope: !11)
!26 = !DILocation(line: 7, column: 24, scope: !11)
!27 = !{!28, !28, i64 0}
!28 = !{!"long long", !29, i64 0}
!29 = !{!"omnipotent char", !30, i64 0}
!30 = !{!"Simple C/C++ TBAA"}
!31 = !DILocation(line: 7, column: 3, scope: !11)
!32 = !DILocation(line: 7, column: 22, scope: !11)
!33 = !DILocation(line: 8, column: 10, scope: !11)
!34 = !DILocation(line: 8, column: 3, scope: !11)
!35 = !DILocation(line: 9, column: 1, scope: !11)
