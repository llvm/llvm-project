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
  %2 = load %a** undef, align 8, !dbg !12
  %3 = bitcast %a* %2 to %a*, !dbg !12
  call void @llvm.dbg.value(metadata %a* %3, i64 0, metadata !6, metadata !{!"0x102"}), !dbg !12
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

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = !{!"0x11\0012\00Apple clang version\001\00\000\00\001", !20, !21, !21, !18, null,  null} ; [ DW_TAG_compile_unit ]
!1 = !{!"0x2e\00foo\00\00\002\000\001\000\006\00256\001\000", !20, !2, !3, null, %a* ()* @test, null, null, !19} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [foo]
!2 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !20, !2, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !0} ; [ DW_TAG_base_type ]
!6 = !{!"0x101\00i\0016777218\000", !1, !2, !5} ; [ DW_TAG_arg_variable ]
!7 = !{!"0x101\00c\0033554434\000", !1, !2, !8} ; [ DW_TAG_arg_variable ]
!8 = !{!"0xf\00\000\0064\0064\000\000", null, !0, !9} ; [ DW_TAG_pointer_type ]
!9 = !{!"0x24\00char\000\008\008\000\000\006", null, !0} ; [ DW_TAG_base_type ]
!10 = !{!"0x100\00a\003\000", !11, !2, !9} ; [ DW_TAG_auto_variable ]
!11 = !{!"0xb\002\0025\000", !20, !1} ; [ DW_TAG_lexical_block ]
!12 = !MDLocation(line: 2, column: 13, scope: !1)
!18 = !{!1}
!19 = !{!6, !7, !10}
!20 = !{!"a.c", !"/private/tmp"}
!21 = !{i32 0}
!22 = !{i32 1, !"Debug Info Version", i32 2}
