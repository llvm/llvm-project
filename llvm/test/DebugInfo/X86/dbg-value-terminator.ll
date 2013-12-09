; RUN: llc -mtriple=x86_64-apple-macosx < %s -verify-machineinstrs | FileCheck %s
;
; PR16143: MachineOperand::setIsKill(bool): Assertion
;
; verify-machineinstrs should ensure that DEBUG_VALUEs go before the
; terminator.
;
; CHECK-LABEL: test:
; CHECK: ##DEBUG_VALUE: i
%a = type { i32, i32 }

define hidden fastcc %a* @test() #1 {
entry:
  %0 = icmp eq %a* undef, null, !dbg !1
  br i1 %0, label %"14", label %return, !dbg !1

"14":                                             ; preds = %"8"
  br i1 undef, label %"25", label %"21", !dbg !1

"21":                                             ; preds = %"14"
  br i1 undef, label %may_unswitch_on.exit, label %"6.i", !dbg !1

"6.i":                                            ; preds = %"21"
  br i1 undef, label %"10.i", label %may_unswitch_on.exit, !dbg !1

"10.i":                                           ; preds = %"6.i"
  br i1 undef, label %may_unswitch_on.exit, label %"12.i", !dbg !1

"12.i":                                           ; preds = %"10.i"
  br i1 undef, label %"4.i.i", label %"3.i.i", !dbg !1

"3.i.i":                                          ; preds = %"12.i"
  br i1 undef, label %"4.i.i", label %VEC_edge_base_index.exit.i, !dbg !1

"4.i.i":                                          ; preds = %"3.i.i", %"12.i"
  unreachable, !dbg !1

VEC_edge_base_index.exit.i:                       ; preds = %"3.i.i"
  br i1 undef, label %may_unswitch_on.exit, label %"16.i", !dbg !1

"16.i":                                           ; preds = %VEC_edge_base_index.exit.i
  br i1 undef, label %"4.i6.i", label %"3.i5.i", !dbg !1

"3.i5.i":                                         ; preds = %"16.i"
  br i1 undef, label %VEC_edge_base_index.exit7.i, label %"4.i6.i", !dbg !1

"4.i6.i":                                         ; preds = %"3.i5.i", %"16.i"
  unreachable, !dbg !1

VEC_edge_base_index.exit7.i:                      ; preds = %"3.i5.i"
  br i1 undef, label %may_unswitch_on.exit, label %"21.i", !dbg !1

"21.i":                                           ; preds = %VEC_edge_base_index.exit7.i
  br i1 undef, label %may_unswitch_on.exit, label %"23.i", !dbg !1

"23.i":                                           ; preds = %"21.i"
  br i1 undef, label %may_unswitch_on.exit, label %"26.i", !dbg !1

"26.i":                                           ; preds = %"34.i", %"23.i"
  %1 = icmp eq i32 undef, 9, !dbg !1
  br i1 %1, label %"34.i", label %"28.i", !dbg !1

"28.i":                                           ; preds = %"26.i"
  unreachable

"34.i":                                           ; preds = %"26.i"
  br i1 undef, label %"26.i", label %"36.i", !dbg !1

"36.i":                                           ; preds = %"34.i"
  br i1 undef, label %"37.i", label %"38.i", !dbg !1

"37.i":                                           ; preds = %"36.i"
  br label %"38.i", !dbg !1

"38.i":                                           ; preds = %"37.i", %"36.i"
  br i1 undef, label %"39.i", label %"45.i", !dbg !1

"39.i":                                           ; preds = %"38.i"
  br i1 undef, label %"41.i", label %may_unswitch_on.exit, !dbg !1

"41.i":                                           ; preds = %"39.i"
  br i1 undef, label %may_unswitch_on.exit, label %"42.i", !dbg !1

"42.i":                                           ; preds = %"41.i"
  br i1 undef, label %may_unswitch_on.exit, label %"44.i", !dbg !1

"44.i":                                           ; preds = %"42.i"
  %2 = load %a** undef, align 8, !dbg !1
  %3 = bitcast %a* %2 to %a*, !dbg !1
  call void @llvm.dbg.value(metadata !{%a* %3}, i64 0, metadata !6), !dbg !12
  br label %may_unswitch_on.exit, !dbg !1

"45.i":                                           ; preds = %"38.i"
  unreachable

may_unswitch_on.exit:                             ; preds = %"44.i", %"42.i", %"41.i", %"39.i", %"23.i", %"21.i", %VEC_edge_base_index.exit7.i, %VEC_edge_base_index.exit.i, %"10.i", %"6.i", %"21"
  %4 = phi %a* [ %3, %"44.i" ], [ null, %"6.i" ], [ null, %"10.i" ], [ null, %VEC_edge_base_index.exit7.i ], [ null, %VEC_edge_base_index.exit.i ], [ null, %"21.i" ], [ null, %"23.i" ], [ null, %"39.i" ], [ null, %"42.i" ], [ null, %"41.i" ], [ null, %"21" ]
  br label %return

"25":                                             ; preds = %"14"
  unreachable

"return":
  %result = phi %a* [ null, %entry ], [ %4, %may_unswitch_on.exit ]
  ret %a* %result, !dbg !1
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind uwtable }

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = metadata !{i32 786449, metadata !20, i32 12, metadata !"Apple clang version", i1 true, metadata !"", i32 0, metadata !21, metadata !21, metadata !18, null,  null, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 786478, metadata !20, metadata !2, metadata !"foo", metadata !"", metadata !"", i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, %a* ()* @test, null, null, metadata !19, i32 0} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!2 = metadata !{i32 786473, metadata !20} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786453, metadata !20, metadata !2, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, null, metadata !0, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 786689, metadata !1, metadata !"i", metadata !2, i32 16777218, metadata !5, i32 0, null} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 786689, metadata !1, metadata !"c", metadata !2, i32 33554434, metadata !8, i32 0, null} ; [ DW_TAG_arg_variable ]
!8 = metadata !{i32 786447, null, metadata !0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ]
!9 = metadata !{i32 786468, null, metadata !0, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 786688, metadata !11, metadata !"a", metadata !2, i32 3, metadata !9, i32 0, null} ; [ DW_TAG_auto_variable ]
!11 = metadata !{i32 786443, metadata !20, metadata !1, i32 2, i32 25, i32 0} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 2, i32 13, metadata !1, null}
!18 = metadata !{metadata !1}
!19 = metadata !{metadata !6, metadata !7, metadata !10}
!20 = metadata !{metadata !"a.c", metadata !"/private/tmp"}
!21 = metadata !{i32 0}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
