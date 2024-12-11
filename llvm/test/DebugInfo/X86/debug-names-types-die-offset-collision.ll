; UNSUPPORTED: system-windows

;; This test checks that DW_IDX_parent is generated correctly when there is DIE relative offset collision between CU and TU.

; RUN: llc -mtriple=x86_64 -generate-type-units -dwarf-version=5 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info -debug-names %t | FileCheck %s

; CHECK: .debug_info contents:
; CHECK:        0x00000023:   DW_TAG_namespace
; CHECK-NEXT:                   DW_AT_name  ("B")
; CHECK:        0x00000023:   DW_TAG_subprogram
; CHECK-NEXT:                   DW_AT_low_pc
; CHECK-NEXT:                   DW_AT_high_pc
; CHECK-NEXT:                   DW_AT_frame_base
; CHECK-NEXT:                   DW_AT_linkage_name  ("_Z9get_statev")
; CHECK-NEXT:                   DW_AT_name  ("get_state")

; CHECK: .debug_names contents:
; CHECK:  String: {{.*}} "B"
; CHECK:        Entry @ [[ENTRY:0x[0-9a-f]*]]
; CHECK:  String: {{.*}} "State"
; CHECK:        Entry @ 0xd3 {
; CHECK:          Abbrev: 0x4
; CHECK:          Tag: DW_TAG_structure_type
; CHECK:          DW_IDX_type_unit: 0x00
; CHECK:          DW_IDX_die_offset: 0x00000025
; CHECK:          DW_IDX_parent: Entry @ [[ENTRY:0x[0-9a-f]*]]
; CHECK:        }


;; namespace B { struct State { class InnerState{}; }; }
;; B::State::InnerState get_state() { return B::State::InnerState(); }
;; clang++ main.cpp -g2 -O0 -fdebug-types-section -gpubnames

; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z9get_statev() #0 !dbg !10 {
entry:
  ret void, !dbg !17
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!1 = !DIFile(filename: "main.cpp", directory: "/folder", checksumkind: CSK_MD5, checksum: "a84fe2e4ecb77633f6c33f3b6833b9e7")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 19.0.0git"}
!10 = distinct !DISubprogram(name: "get_state", linkageName: "_Z9get_statev", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "InnerState", scope: !14, file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !16, identifier: "_ZTSN1B5State10InnerStateE")
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "State", scope: !15, file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !16, identifier: "_ZTSN1B5StateE")
!15 = !DINamespace(name: "B", scope: null)
!16 = !{}
!17 = !DILocation(line: 2, column: 36, scope: !10)
