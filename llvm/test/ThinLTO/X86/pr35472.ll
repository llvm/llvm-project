; Test to make sure that lazily loaded debug location scope metadata is
; handled properly. Note that we need to have the DILexicalScope !34
; referenced from multiple function's debug locs for this to be in the
; lazily loaded module level metadata block.

; RUN: opt -module-hash -module-summary %s -o %t1.bc
; RUN: opt -module-hash -module-summary %p/Inputs/pr35472.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=run %t1.bc %t2.bc -exported-symbol=_Z5Alphav
; RUN: llvm-nm %t1.bc.thinlto.o | FileCheck %s -check-prefix=ThinLTOa
; RUN: llvm-nm %t2.bc.thinlto.o | FileCheck %s -check-prefix=ThinLTOb

;; Re-run with "new" debug-info mode, checking that we load / convert / emit
;; the dbg.declares below correctly.
; RUN: llvm-lto -thinlto-action=run %t1.bc %t2.bc -exported-symbol=_Z5Alphav --try-experimental-debuginfo-iterators
; RUN: llvm-nm %t1.bc.thinlto.o | FileCheck %s -check-prefix=ThinLTOa
; RUN: llvm-nm %t2.bc.thinlto.o | FileCheck %s -check-prefix=ThinLTOb

; ThinLTOa-DAG: T _Z5Bravov
; ThinLTOa-DAG: W _ZN4EchoD2Ev
; ThinLTOb-DAG: T _Z5Alphav

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Delta = type { %struct.Charlie }
%struct.Charlie = type { i32 }
%struct.Echo = type { %struct.Charlie }

$_ZN4EchoD2Ev = comdat any
$_ZN5DeltaD2Ev = comdat any

define void @_Z5Bravov() !dbg !7 {
  %Hotel = alloca %struct.Delta, align 4
  %India = alloca %struct.Echo, align 4
  call void @llvm.dbg.declare(metadata ptr %Hotel, metadata !10, metadata !DIExpression()), !dbg !22
  call void @_ZN4EchoD2Ev(ptr %India), !dbg !28
  ret void, !dbg !28
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define linkonce_odr void @_ZN4EchoD2Ev(ptr %this) unnamed_addr comdat align 2 {
  %this.addr.i = alloca ptr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr.i, metadata !29, metadata !DIExpression()), !dbg !32
  %this1.i = load ptr, ptr %this.addr.i, align 8
  ret void
}

define linkonce_odr void @_ZN5DeltaD2Ev(ptr %this) unnamed_addr comdat align 2 !dbg !36 {
  %this.addr.i = alloca ptr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr.i, metadata !29, metadata !DIExpression()), !dbg !41
  %this1.i = load ptr, ptr %this.addr.i, align 8
  ret void
}

!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 321056)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/home/sunil/185335/302")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!7 = distinct !DISubprogram(name: "Bravo", linkageName: "_Z5Bravov", scope: !1, file: !1, line: 17, type: !8, isLocal: false, isDefinition: true, scopeLine: 17, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "Hotel", scope: !7, file: !1, line: 18, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Delta", file: !1, line: 6, size: 32, elements: !12, identifier: "_ZTS5Delta")
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "Foxtrot", scope: !11, file: !1, line: 7, baseType: !14, size: 32)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Charlie", file: !1, line: 1, size: 32, elements: !15, identifier: "_ZTS7Charlie")
!15 = !{!16, !18}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "Golf", scope: !14, file: !1, line: 3, baseType: !17, size: 32)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DISubprogram(name: "~Charlie", scope: !14, file: !1, line: 2, type: !19, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !21}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DILocation(line: 18, column: 11, scope: !7)
!24 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Echo", file: !1, line: 10, size: 32, elements: !25, identifier: "_ZTS4Echo")
!25 = !{!26}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "Foxtrot", scope: !24, file: !1, line: 11, baseType: !14, size: 32)
!28 = !DILocation(line: 20, column: 1, scope: !7)
!29 = !DILocalVariable(name: "this", arg: 1, scope: !30, type: !31, flags: DIFlagArtificial | DIFlagObjectPointer)
!30 = distinct !DISubprogram(name: "~Charlie", linkageName: "_ZN7CharlieD2Ev", scope: !14, file: !1, line: 2, type: !19, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !18)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!32 = !DILocation(line: 0, scope: !30)
!33 = !DILocation(line: 2, column: 53, scope: !34)
!34 = distinct !DILexicalBlock(scope: !30, file: !1, line: 2, column: 51)
!36 = distinct !DISubprogram(name: "~Delta", linkageName: "_ZN5DeltaD2Ev", scope: !11, file: !1, line: 6, type: !37, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !40)
!37 = !DISubroutineType(types: !38)
!38 = !{null, !39}
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!40 = !DISubprogram(name: "~Delta", scope: !11, type: !37, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!41 = !DILocation(line: 0, scope: !30, inlinedAt: !42)
!42 = distinct !DILocation(line: 6, column: 8, scope: !43)
!43 = distinct !DILexicalBlock(scope: !36, file: !1, line: 6, column: 8)
!48 = !DILocation(line: 2, column: 53, scope: !34, inlinedAt: !42)

;----------------------------------------------------------------------------------------------
; Compiled from following two source files with 'clang++ -S --std=c++11 -O0 -g -flto=thin' 
; struct Charlie {
;     __attribute__((__always_inline__)) ~Charlie() { Golf = 0; }
;     int Golf;
; };
; 
; struct Delta {
;     Charlie Foxtrot;
; };
; 
; struct Echo {
;     Charlie Foxtrot;
;     __attribute__((nodebug)) ~Echo() = default;
; };
; 
; extern void Bravo();
; 
; void Bravo() {
;     Delta Hotel;
;     Echo India;
; }
; -----------------------------
; extern void Bravo();
; extern void Alpha();
; void Alpha() { Bravo(); }

