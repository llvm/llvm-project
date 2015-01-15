; RUN: llc -mtriple=x86_64-apple-darwin10 < %s | FileCheck %s

%struct.a = type { i32 }

define i32 @bar(%struct.a* nocapture %b) nounwind ssp {
entry:
  tail call void @llvm.dbg.value(metadata %struct.a* %b, i64 0, metadata !6, metadata !{!"0x102"}), !dbg !13
  %tmp1 = getelementptr inbounds %struct.a* %b, i64 0, i32 0, !dbg !14
  %tmp2 = load i32* %tmp1, align 4, !dbg !14
  tail call void @llvm.dbg.value(metadata i32 %tmp2, i64 0, metadata !11, metadata !{!"0x102"}), !dbg !14
  %call = tail call i32 (...)* @foo(i32 %tmp2) nounwind , !dbg !18
  %add = add nsw i32 %tmp2, 1, !dbg !19
  ret i32 %add, !dbg !19
}

declare i32 @foo(...) 

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!24}

!0 = !{!"0x2e\00bar\00bar\00\005\000\001\000\006\00256\001\000", !22, !1, !3, null, i32 (%struct.a*)* @bar, null, null, !21} ; [ DW_TAG_subprogram ] [line 5] [def] [scope 0] [bar]
!1 = !{!"0x29", !22} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 122997)\001\00\000\00\001", !22, !23, !23, !20, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !22, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!6 = !{!"0x101\00b\005\000", !0, !1, !7} ; [ DW_TAG_arg_variable ]
!7 = !{!"0xf\00\000\0064\0064\000\000", null, !2, !8} ; [ DW_TAG_pointer_type ]
!8 = !{!"0x13\00a\001\0032\0032\000\000\000", !22, !2, null, !9, null, null, null} ; [ DW_TAG_structure_type ] [a] [line 1, size 32, align 32, offset 0] [def] [from ]
!9 = !{!10}
!10 = !{!"0xd\00c\002\0032\0032\000\000", !22, !1, !5} ; [ DW_TAG_member ]
!11 = !{!"0x100\00x\006\000", !12, !1, !5} ; [ DW_TAG_auto_variable ]
!12 = !{!"0xb\005\0022\000", !22, !0} ; [ DW_TAG_lexical_block ]
!13 = !MDLocation(line: 5, column: 19, scope: !0)
!14 = !MDLocation(line: 6, column: 14, scope: !12)
!18 = !MDLocation(line: 7, column: 2, scope: !12)
!19 = !MDLocation(line: 8, column: 2, scope: !12)
!20 = !{!0}
!21 = !{!6, !11}
!22 = !{!"bar.c", !"/private/tmp"}
!23 = !{i32 0}

; Check that variable bar:b value range is appropriately truncated in debug info.
; The variable is in %rdi which is clobbered by 'movl %ebx, %edi'
; Here Ltmp7 is the end of the location range.

;CHECK: .loc	1 7 2
;CHECK: movl
;CHECK-NEXT: [[CLOBBER:Ltmp[0-9]*]]

;CHECK:Ldebug_loc0:
;CHECK-NEXT: Lset{{.*}} =
;CHECK-NEXT:	.quad
;CHECK-NEXT: [[CLOBBER_OFF:Lset.*]] = [[CLOBBER]]-{{.*}}
;CHECK-NEXT:	.quad	[[CLOBBER_OFF]]
;CHECK-NEXT: Lset{{.*}} = Ltmp{{.*}}-Ltmp{{.*}}
;CHECK-NEXT:    .short  Lset
;CHECK-NEXT: Ltmp
;CHECK-NEXT:	.byte	85 ## DW_OP_reg
;CHECK-NEXT: Ltmp
;CHECK-NEXT:	.quad	0
;CHECK-NEXT:	.quad	0
!24 = !{i32 1, !"Debug Info Version", i32 2}
