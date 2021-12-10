; RUN: %llc_dwarf -filetype=obj -o - %s | llvm-dwarfdump -| FileCheck --implicit-check-not=DW_TAG --implicit-check-not=NULL %s
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_language	(DW_LANG_Rust)
; CHECK:   DW_TAG_namespace
; CHECK:     DW_TAG_enumeration_type
; CHECK:     DW_AT_name	("E")
; CHECK:       DW_TAG_enumerator
; CHECK:       DW_TAG_enumerator
; CHECK:       DW_TAG_subprogram
; CHECK:         DW_AT_name	("f")
; CHECK:         DW_TAG_formal_parameter
; CHECK:         NULL
; CHECK:       NULL
; CHECK:     NULL
; CHECK:   DW_TAG_pointer_type
; CHECK:   NULL

; This file comes from rustc output, with the input program
;         pub enum E { A, B }
;         impl E {
;             pub fn f(&self) {}
;         }
; compiled with `rustc --crate-type=lib a.rs --emit llvm-ir -g` and
; copying the resulting `a.ll` file to here. This was done with rustc
; at nightly from 2021-09-28 (git 8f8092cc3), but rustc 1.57 should
; produce similar or identical output.

; ModuleID = 'a.a146b597-cgu.0'
source_filename = "a.a146b597-cgu.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

; a::E::f
; Function Attrs: uwtable
define void @_ZN1a1E1f17h4fcb50ce732fb2a7E(i8* align 1 dereferenceable(1) %self) unnamed_addr #0 !dbg !13 {
start:
  %self.dbg.spill = alloca i8*, align 8
  store i8* %self, i8** %self.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata i8** %self.dbg.spill, metadata !19, metadata !DIExpression()), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { uwtable "frame-pointer"="all" "probe-stack"="__rust_probestack" "target-cpu"="core2" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!3}

!0 = !{i32 7, !"PIC Level", i32 2}
!1 = !{i32 2, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !4, producer: "clang LLVM (rustc version 1.57.0-nightly (8f8092cc3 2021-09-28))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5)
!4 = !DIFile(filename: "a.rs/@/a.a146b597-cgu.0", directory: "/Users/augie")
!5 = !{!6}
!6 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", scope: !8, file: !7, baseType: !9, size: 8, align: 8, flags: DIFlagEnumClass, elements: !10)
!7 = !DIFile(filename: "<unknown>", directory: "")
!8 = !DINamespace(name: "a", scope: null)
!9 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!10 = !{!11, !12}
!11 = !DIEnumerator(name: "A", value: 0)
!12 = !DIEnumerator(name: "B", value: 1)
!13 = distinct !DISubprogram(name: "f", linkageName: "_ZN1a1E1f17h4fcb50ce732fb2a7E", scope: !6, file: !14, line: 3, type: !15, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, templateParams: !20, retainedNodes: !18)
!14 = !DIFile(filename: "a.rs", directory: "/Users/augie", checksumkind: CSK_MD5, checksum: "ab4ce84c27ef6fd0be1ef78e8131faa8")
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&E", baseType: !6, size: 64, align: 64, dwarfAddressSpace: 0)
!18 = !{!19}
!19 = !DILocalVariable(name: "self", arg: 1, scope: !13, file: !14, line: 3, type: !17)
!20 = !{}
!21 = !DILocation(line: 3, column: 14, scope: !13)
!22 = !DILocation(line: 3, column: 23, scope: !13)
