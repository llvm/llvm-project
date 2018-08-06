; RUN: llc -mtriple=x86_64-apple-macosx < %s -verify-machineinstrs | FileCheck %s
;
; PR16143: MachineOperand::setIsKill(bool): Assertion
;
; verify-machineinstrs should ensure that DEBUG_VALUEs go before the
; terminator.
;
; CHECK-LABEL: test:
; CHECK: ##DEBUG_VALUE: foo:i

%a = type { i32, i32 }

define hidden fastcc %a* @test() #1 !dbg !1 {
entry:
  %0 = icmp eq %a* undef, null, !dbg !12
  br i1 %0, label %"14", label %return, !dbg !12

"14":                                             ; preds = %"8"
  br i1 undef, label %"25", label %"21", !dbg !12

"21":                                             ; preds = %"14"
  br i1 undef, label %may_unswitch_on.exit, label %"6.i", !dbg !12

"6.i":                                            ; preds = %"21"
  br i1 undef, label %"10.i", label %may_unswitch_on.exit, !dbg !12

"10.i":                                           ; preds = %"6.i"
  br i1 undef, label %may_unswitch_on.exit, label %"12.i", !dbg !12

"12.i":                                           ; preds = %"10.i"
  br i1 undef, label %"4.i.i", label %"3.i.i", !dbg !12

"3.i.i":                                          ; preds = %"12.i"
  br i1 undef, label %"4.i.i", label %VEC_edge_base_index.exit.i, !dbg !12

"4.i.i":                                          ; preds = %"3.i.i", %"12.i"
  unreachable, !dbg !12

VEC_edge_base_index.exit.i:                       ; preds = %"3.i.i"
  br i1 undef, label %may_unswitch_on.exit, label %"16.i", !dbg !12

"16.i":                                           ; preds = %VEC_edge_base_index.exit.i
  br i1 undef, label %"4.i6.i", label %"3.i5.i", !dbg !12

"3.i5.i":                                         ; preds = %"16.i"
  br i1 undef, label %VEC_edge_base_index.exit7.i, label %"4.i6.i", !dbg !12

"4.i6.i":                                         ; preds = %"3.i5.i", %"16.i"
  unreachable, !dbg !12

VEC_edge_base_index.exit7.i:                      ; preds = %"3.i5.i"
  br i1 undef, label %may_unswitch_on.exit, label %"21.i", !dbg !12

"21.i":                                           ; preds = %VEC_edge_base_index.exit7.i
  br i1 undef, label %may_unswitch_on.exit, label %"23.i", !dbg !12

"23.i":                                           ; preds = %"21.i"
  br i1 undef, label %may_unswitch_on.exit, label %"26.i", !dbg !12

"26.i":                                           ; preds = %"34.i", %"23.i"
  %1 = icmp eq i32 undef, 9, !dbg !12
  br i1 %1, label %"34.i", label %"28.i", !dbg !12

"28.i":                                           ; preds = %"26.i"
  unreachable

"34.i":                                           ; preds = %"26.i"
  br i1 undef, label %"26.i", label %"36.i", !dbg !12

"36.i":                                           ; preds = %"34.i"
  br i1 undef, label %"37.i", label %"38.i", !dbg !12

"37.i":                                           ; preds = %"36.i"
  br label %"38.i", !dbg !12

"38.i":                                           ; preds = %"37.i", %"36.i"
  br i1 undef, label %"39.i", label %"45.i", !dbg !12

"39.i":                                           ; preds = %"38.i"
  br i1 undef, label %"41.i", label %may_unswitch_on.exit, !dbg !12

"41.i":                                           ; preds = %"39.i"
  br i1 undef, label %may_unswitch_on.exit, label %"42.i", !dbg !12

"42.i":                                           ; preds = %"41.i"
  br i1 undef, label %may_unswitch_on.exit, label %"44.i", !dbg !12

"44.i":                                           ; preds = %"42.i"
  %2 = load %a*, %a** undef, align 8, !dbg !12
  %3 = bitcast %a* %2 to %a*, !dbg !12
  call void @llvm.dbg.value(metadata %a* %3, metadata !6, metadata !DIExpression(DW_OP_deref)), !dbg !12
  br label %may_unswitch_on.exit, !dbg !12

"45.i":                                           ; preds = %"38.i"
  unreachable

may_unswitch_on.exit:                             ; preds = %"44.i", %"42.i", %"41.i", %"39.i", %"23.i", %"21.i", %VEC_edge_base_index.exit7.i, %VEC_edge_base_index.exit.i, %"10.i", %"6.i", %"21"
  %4 = phi %a* [ %3, %"44.i" ], [ null, %"6.i" ], [ null, %"10.i" ], [ null, %VEC_edge_base_index.exit7.i ], [ null, %VEC_edge_base_index.exit.i ], [ null, %"21.i" ], [ null, %"23.i" ], [ null, %"39.i" ], [ null, %"42.i" ], [ null, %"41.i" ], [ null, %"21" ]
  br label %return

"25":                                             ; preds = %"14"
  unreachable

"return":
  %result = phi %a* [ null, %entry ], [ %4, %may_unswitch_on.exit ]
  ret %a* %result, !dbg !12
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind uwtable }

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "Apple clang version", isOptimized: true, emissionKind: FullDebug, file: !20, enums: !21, retainedTypes: !21, imports:  null)
!1 = distinct !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, file: !20, scope: !2, type: !3, retainedNodes: !19)
!2 = !DIFile(filename: "a.c", directory: "/private/tmp")
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "i", line: 2, arg: 1, scope: !1, file: !2, type: !5)
!7 = !DILocalVariable(name: "c", line: 2, arg: 2, scope: !1, file: !2, type: !8)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, scope: !0, baseType: !9)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!10 = !DILocalVariable(name: "a", line: 3, scope: !11, file: !2, type: !9)
!11 = distinct !DILexicalBlock(line: 2, column: 25, file: !20, scope: !1)
!12 = !DILocation(line: 2, column: 13, scope: !1)
!19 = !{!6, !7, !10}
!20 = !DIFile(filename: "a.c", directory: "/private/tmp")
!21 = !{}
!22 = !{i32 1, !"Debug Info Version", i32 3}
