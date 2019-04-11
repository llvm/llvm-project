; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump -v - | FileCheck %s
; This tests a fragment that partially covers subregister compositions.
;
; Our fragment is 96 bits long and lies in a 128-bit register, which
; in turn has to be composed out of its two 64-bit subregisters.

; CHECK: .debug_info
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name {{.*}}"subscript.get"
; CHECK:  DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location [DW_FORM_sec_offset]	({{.*}}
; CHECK-NEXT:  [0x{{.*}}, 0x{{.*}}): DW_OP_regx D16, DW_OP_piece 0x8, DW_OP_regx D17, DW_OP_piece 0x4
; CHECK-NEXT:  [0x{{.*}}, 0x{{.*}}): DW_OP_regx D16, DW_OP_piece 0x8, DW_OP_regx D17, DW_OP_piece 0x4

; FIXME: The second location list entry should not be emitted.
;
; The input to LiveDebugValues is:
;
; bb.0.entry:
;   [...]
;   Bcc %bb.2, 1, killed $cpsr, debug-location !10; simd.swift:5900:12
; bb.1:
;   [...]
;   DBG_VALUE $q8, $noreg, !"self", !DIExpression(DW_OP_LLVM_fragment, 0, 96)
;   B %bb.3
; bb.2.select.false:
;   [...]
;   DBG_VALUE $q8, $noreg, !"self", !DIExpression(DW_OP_LLVM_fragment, 0, 96)
; bb.3.select.end:
;   [...]
;
; The two DBG_VALUEs in the blocks describe different fragments of the
; variable. However, LiveDebugValues is not aware of fragments, so it will
; incorrectly insert a copy of the first DBG_VALUE in bb.3.select.end, since
; the debug values in its predecessor blocks are described by the same
; register.

source_filename = "simd.ll"
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv7-apple-ios7.0"

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

define <3 x float> @_TFV4simd8float2x3g9subscriptFSiVS_6float3(i32, <3 x float>, <3 x float>) !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata <3 x float> %1, metadata !8, metadata !9), !dbg !10
  tail call void @llvm.dbg.value(metadata <3 x float> %2, metadata !8, metadata !11), !dbg !10
  %3 = icmp eq i32 %0, 0, !dbg !12
  br i1 %3, label %7, label %4, !dbg !12

; <label>:4:                                      ; preds = %entry
  %5 = icmp eq i32 %0, 1, !dbg !15
  br i1 %5, label %7, label %6, !dbg !15

; <label>:6:                                      ; preds = %4
  unreachable, !dbg !17

; <label>:7:                                      ; preds = %4, %entry
  %8 = phi <3 x float> [ %1, %entry ], [ %2, %4 ], !dbg !18
  ret <3 x float> %8, !dbg !18
}

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Swift", isOptimized: false, runtimeVersion: 3, emissionKind: FullDebug, enums: !2, imports: !2)
!1 = !DIFile(filename: "simd.swift", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "subscript.get", linkageName: "_TFV4simd8float2x3g9subscriptFSiVS_6float3", scope: !6, file: !1, type: !7, isLocal: false, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !2)
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "float2x3", scope: !0, file: !1, line: 5824, size: 256, align: 128, elements: !2, runtimeLang: DW_LANG_Swift, identifier: "_TtV4simd8float2x3")
!7 = !DISubroutineType(types: !2)
!8 = !DILocalVariable(name: "self", arg: 2, scope: !5, file: !1, line: 5897, type: !6, flags: DIFlagArtificial)
!9 = !DIExpression(DW_OP_LLVM_fragment, 0, 96)
!10 = !DILocation(line: 5897, column: 5, scope: !5)
!11 = !DIExpression(DW_OP_LLVM_fragment, 96, 96)
!12 = !DILocation(line: 5900, column: 12, scope: !13)
!13 = distinct !DILexicalBlock(scope: !14, file: !1, line: 5898, column: 7)
!14 = distinct !DILexicalBlock(scope: !5, file: !1, line: 5897, column: 9)
!15 = !DILocation(line: 5902, column: 12, scope: !16)
!16 = distinct !DILexicalBlock(scope: !14, file: !1, line: 5898, column: 7)
!17 = !DILocation(line: 0, scope: !5)
!18 = !DILocation(line: 5906, column: 5, scope: !14)
