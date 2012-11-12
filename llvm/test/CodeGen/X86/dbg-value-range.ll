; RUN: llc -mtriple=x86_64-apple-darwin10 < %s | FileCheck %s

%struct.a = type { i32 }

define i32 @bar(%struct.a* nocapture %b) nounwind ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.a* %b}, i64 0, metadata !6), !dbg !13
  %tmp1 = getelementptr inbounds %struct.a* %b, i64 0, i32 0, !dbg !14
  %tmp2 = load i32* %tmp1, align 4, !dbg !14, !tbaa !15
  tail call void @llvm.dbg.value(metadata !{i32 %tmp2}, i64 0, metadata !11), !dbg !14
  %call = tail call i32 (...)* @foo(i32 %tmp2) nounwind , !dbg !18
  %add = add nsw i32 %tmp2, 1, !dbg !19
  ret i32 %add, !dbg !19
}

declare i32 @foo(...) 

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.sp = !{!0}
!llvm.dbg.lv.bar = !{!6, !11}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"bar", metadata !"bar", metadata !"", metadata !1, i32 5, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32 (%struct.a*)* @bar} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"bar.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"bar.c", metadata !"/private/tmp", metadata !"clang version 2.9 (trunk 122997)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !2, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 590081, metadata !0, metadata !"b", metadata !1, i32 5, metadata !7, i32 0} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 589839, metadata !2, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{i32 589843, metadata !2, metadata !"a", metadata !1, i32 1, i64 32, i64 32, i32 0, i32 0, i32 0, metadata !9, i32 0, i32 0} ; [ DW_TAG_structure_type ]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 589837, metadata !1, metadata !"c", metadata !1, i32 2, i64 32, i64 32, i64 0, i32 0, metadata !5} ; [ DW_TAG_member ]
!11 = metadata !{i32 590080, metadata !12, metadata !"x", metadata !1, i32 6, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!12 = metadata !{i32 589835, metadata !0, i32 5, i32 22, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!13 = metadata !{i32 5, i32 19, metadata !0, null}
!14 = metadata !{i32 6, i32 14, metadata !12, null}
!15 = metadata !{metadata !"int", metadata !16}
!16 = metadata !{metadata !"omnipotent char", metadata !17}
!17 = metadata !{metadata !"Simple C/C++ TBAA", null}
!18 = metadata !{i32 7, i32 2, metadata !12, null}
!19 = metadata !{i32 8, i32 2, metadata !12, null}

; Check that variable bar:b value range is appropriately trucated in debug info.
; The variable is in %rdi which is clobbered by 'movl %ebx, %edi'
; Here Ltmp7 is the end of the location range.

;CHECK: .loc	1 7 2
;CHECK: movl
;CHECK-NEXT: [[CLOBBER:Ltmp[0-9]*]]

;CHECK:Ldebug_loc0:
;CHECK-NEXT:	.quad
;CHECK-NEXT:	.quad	[[CLOBBER]]
;CHECK-NEXT: Lset{{.*}} = Ltmp{{.*}}-Ltmp{{.*}}
;CHECK-NEXT:    .short  Lset
;CHECK-NEXT: Ltmp
;CHECK-NEXT:	.byte	85
;CHECK-NEXT: Ltmp
;CHECK-NEXT:	.quad	0
;CHECK-NEXT:	.quad	0
