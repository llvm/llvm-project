; RUN: llc -mtriple=bpfel -filetype=obj %s -o %t.o
; RUN: llvm-objcopy --dump-section=.BTF=%t.btf %t.o
; RUN: %python %p/print_btf.py %t.btf | FileCheck %s --implicit-check-not="'wide"

; Source code:
;   typedef int word;
;   enum small : word { positive = 42 } value;
;   enum wide : unsigned __int128 { big } wide_value;
;   enum wide_forward : unsigned __int128;
;   enum wide_forward wide_decl;
;   enum forward64 : unsigned long long;
;   enum forward64 keep_forward;
;   enum opaque *p;
;   enum legacy { K = 0xf000000000000000ULL } legacy_value;
; Compilation flag:
;   clang -target bpf -std=c23 -O2 -g -S -emit-llvm enum-width.c
; Remove legacy's baseType to match the Clang 3.4 metadata in
; llvm/test/DebugInfo/Generic/enum.ll.

; CHECK: [[SMALL:[0-9]+]]] ENUM 'small' encoding=SIGNED size=4 vlen=1
; CHECK-NEXT: 'positive' val=42
; CHECK: VAR 'value' type_id=[[SMALL]],
; CHECK: [[FWD:[0-9]+]]] ENUM 'forward64' encoding=UNSIGNED size=8 vlen=0
; CHECK: VAR 'keep_forward' type_id=[[FWD]],
; CHECK: [[PTR:[0-9]+]]] PTR '(anon)' type_id=[[OPAQUE:[0-9]+]]
; CHECK: {{\[}}[[OPAQUE]]] ENUM 'opaque' encoding=UNSIGNED size=4 vlen=0
; CHECK: VAR 'p' type_id=[[PTR]],
; CHECK: ENUM64 'legacy' encoding=UNSIGNED size=8 vlen=1
; CHECK-NEXT: 'K' val=17293822569102704640LL
; CHECK: DATASEC '.bss' size=0 vlen=4

@value = global i32 0, !dbg !2
@wide_value = global i128 0, !dbg !7
@wide_decl = global i128 0, !dbg !11
@keep_forward = global i64 0, !dbg !14
@p = global ptr null, !dbg !17
@legacy_value = global i64 0, !dbg !21

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug, globals: !{!2, !7, !11, !14, !17, !21})
!1 = !DIFile(filename: "enum-width.c", directory: "/")
!2 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = distinct !DIGlobalVariable(name: "value", scope: !0, file: !1, line: 2, type: !4, isLocal: false, isDefinition: true)
!4 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "small", file: !1, line: 2, baseType: !5, size: 32, elements: !{!DIEnumerator(name: "positive", value: 42)})
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "word", file: !1, line: 1, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "wide_value", scope: !0, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "wide", file: !1, line: 3, baseType: !10, size: 128, elements: !{!DIEnumerator(name: "big", value: 0, isUnsigned: true)})
!10 = !DIBasicType(name: "unsigned __int128", size: 128, encoding: DW_ATE_unsigned)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "wide_decl", scope: !0, file: !1, line: 5, type: !13, isLocal: false, isDefinition: true)
!13 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "wide_forward", file: !1, line: 4, size: 128, flags: DIFlagFwdDecl)
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression())
!15 = distinct !DIGlobalVariable(name: "keep_forward", scope: !0, file: !1, line: 7, type: !16, isLocal: false, isDefinition: true)
!16 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "forward64", file: !1, line: 6, size: 64, flags: DIFlagFwdDecl)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "p", scope: !0, file: !1, line: 8, type: !19, isLocal: false, isDefinition: true)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "opaque", file: !1, line: 8, flags: DIFlagFwdDecl)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "legacy_value", scope: !0, file: !1, line: 9, type: !23, isLocal: false, isDefinition: true)
!23 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "legacy", file: !1, line: 9, size: 64, elements: !{!DIEnumerator(name: "K", value: 17293822569102704640, isUnsigned: true)})
!24 = !{i32 2, !"Debug Info Version", i32 3}
