; RUN: llc -mtriple=bpfel -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
;
; This IR is hand-written.

; ModuleID = 'ptr-named-2.ll'
source_filename = "ptr-named-2.ll"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpfel-unknown-none"

%struct.TypeExamples = type { ptr, i32, i32, ptr }

@type_examples = internal global %struct.TypeExamples zeroinitializer, align 8, !dbg !0

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!21}

; CHECK-BTF:      [1] STRUCT 'TypeExamples' size=32 vlen=4
; CHECK-BTF-NEXT:         'ptr' type_id=2 bits_offset=0
; CHECK-BTF-NEXT:         'volatile' type_id=4 bits_offset=64
; CHECK-BTF-NEXT:         'const' type_id=5 bits_offset=128
; CHECK-BTF-NEXT:         'restrict_ptr' type_id=6 bits_offset=192
; CHECK-BTF-NEXT: [2] PTR '(anon)' type_id=3
; CHECK-BTF-NEXT: [3] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF-NEXT: [4] VOLATILE '(anon)' type_id=3
; CHECK-BTF-NEXT: [5] CONST '(anon)' type_id=3
; CHECK-BTF-NEXT: [6] RESTRICT '(anon)' type_id=7
; CHECK-BTF-NEXT: [7] PTR '(anon)' type_id=3
; CHECK-BTF-NEXT: [8] VAR 'type_examples' type_id=1, linkage=static
; CHECK-BTF-NEXT: [9] DATASEC '.bss' size=0 vlen=1
; CHECK-BTF-NEXT:         type_id=8 offset=0 size=24

!0 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, globals: !8, splitDebugInlining: false, nameTableKind: None)
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = distinct !DIGlobalVariable(name: "type_examples", scope: !1, file: !6, line: 12, type: !9, isLocal: true, isDefinition: true)
!6 = !DIFile(filename: "ptr-named-2.ll", directory: "/tmp")
!7 = !{}
!8 = !{!0}
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TypeExamples", file: !6, line: 5, size: 256, elements: !10)
!10 = !{!11, !12, !13, !14}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !9, file: !6, line: 6, baseType: !15, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "volatile", scope: !9, file: !6, line: 7, baseType: !17, size: 64, offset: 64)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "const", scope: !9, file: !6, line: 8, baseType: !18, size: 64, offset: 128)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "restrict_ptr", scope: !9, file: !6, line: 9, baseType: !19, size: 64, offset: 192)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*int", baseType: !16, size: 64)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DIDerivedType(tag: DW_TAG_volatile_type, name: "volatile int", baseType: !16)
!18 = !DIDerivedType(tag: DW_TAG_const_type, name: "const int", baseType: !16)
!19 = !DIDerivedType(tag: DW_TAG_restrict_type, name: "*int restrict", baseType: !20)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!21 = !{!"my hand-written IR"}
