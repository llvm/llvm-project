; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O2 -regalloc=basic < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10"

%struct.a = type { i32, %struct.a* }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i8* (%struct.a*)* @bar to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define i8* @bar(%struct.a* %myvar) nounwind optsize noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.a* %myvar}, i64 0, metadata !8)
  %0 = getelementptr inbounds %struct.a* %myvar, i64 0, i32 0, !dbg !28 ; <i32*> [#uses=1]
  %1 = load i32* %0, align 8, !dbg !28            ; <i32> [#uses=1]
  tail call void @foo(i32 %1) nounwind optsize noinline ssp, !dbg !28
  %2 = bitcast %struct.a* %myvar to i8*, !dbg !30 ; <i8*> [#uses=1]
  ret i8* %2, !dbg !30
}

declare void @foo(i32) nounwind optsize noinline ssp

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!38}

!0 = metadata !{i32 786484, i32 0, metadata !1, metadata !"ret", metadata !"ret", metadata !"", metadata !1, i32 7, metadata !3, i1 false, i1 true, null, null} ; [ DW_TAG_variable ]
!1 = metadata !{i32 786473, metadata !36} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !36, i32 1, metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, metadata !"", i32 0, metadata !37, metadata !37, metadata !32, metadata !31,  metadata !31, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786468, metadata !36, metadata !1, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!4 = metadata !{i32 786689, metadata !5, metadata !"x", metadata !1, i32 12, metadata !3, i32 0, null} ; [ DW_TAG_arg_variable ]
!5 = metadata !{i32 786478, metadata !36, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", i32 13, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, void (i32)* @foo, null, null, metadata !33, i32 13} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786453, metadata !36, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !3}
!8 = metadata !{i32 786689, metadata !9, metadata !"myvar", metadata !1, i32 17, metadata !13, i32 0, null} ; [ DW_TAG_arg_variable ]
!9 = metadata !{i32 786478, metadata !36, metadata !1, metadata !"bar", metadata !"bar", metadata !"bar", i32 17, metadata !10, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i8* (%struct.a*)* @bar, null, null, metadata !34, i32 17} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 786453, metadata !36, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{metadata !12, metadata !13}
!12 = metadata !{i32 786447, metadata !36, metadata !1, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!13 = metadata !{i32 786447, metadata !36, metadata !1, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !14} ; [ DW_TAG_pointer_type ]
!14 = metadata !{i32 786451, metadata !36, metadata !1, metadata !"a", i32 2, i64 128, i64 64, i64 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [a] [line 2, size 128, align 64, offset 0] [def] [from ]
!15 = metadata !{metadata !16, metadata !17}
!16 = metadata !{i32 786445, metadata !36, metadata !14, metadata !"c", i32 3, i64 32, i64 32, i64 0, i32 0, metadata !3} ; [ DW_TAG_member ]
!17 = metadata !{i32 786445, metadata !36, metadata !14, metadata !"d", i32 4, i64 64, i64 64, i64 64, i32 0, metadata !13} ; [ DW_TAG_member ]
!18 = metadata !{i32 786689, metadata !19, metadata !"argc", metadata !1, i32 22, metadata !3, i32 0, null} ; [ DW_TAG_arg_variable ]
!19 = metadata !{i32 786478, metadata !36, metadata !1, metadata !"main", metadata !"main", metadata !"main", i32 22, metadata !20, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, null, null, null, metadata !35, i32 22} ; [ DW_TAG_subprogram ]
!20 = metadata !{i32 786453, metadata !36, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !21, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{metadata !3, metadata !3, metadata !22}
!22 = metadata !{i32 786447, metadata !36, metadata !1, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !23} ; [ DW_TAG_pointer_type ]
!23 = metadata !{i32 786447, metadata !36, metadata !1, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !24} ; [ DW_TAG_pointer_type ]
!24 = metadata !{i32 786468, metadata !36, metadata !1, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!25 = metadata !{i32 786689, metadata !19, metadata !"argv", metadata !1, i32 22, metadata !22, i32 0, null} ; [ DW_TAG_arg_variable ]
!26 = metadata !{i32 786688, metadata !27, metadata !"e", metadata !1, i32 23, metadata !14, i32 0, null} ; [ DW_TAG_auto_variable ]
!27 = metadata !{i32 786443, metadata !36, metadata !19, i32 22, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!28 = metadata !{i32 18, i32 0, metadata !29, null}
!29 = metadata !{i32 786443, metadata !36, metadata !9, i32 17, i32 0, i32 1} ; [ DW_TAG_lexical_block ]
!30 = metadata !{i32 19, i32 0, metadata !29, null}
!31 = metadata !{metadata !0}
!32 = metadata !{metadata !5, metadata !9, metadata !19}
!33 = metadata !{metadata !4}
!34 = metadata !{metadata !8}
!35 = metadata !{metadata !18, metadata !25, metadata !26}
!36 = metadata !{metadata !"foo.c", metadata !"/tmp/"}
!37 = metadata !{i32 0}

; The variable bar:myvar changes registers after the first movq.
; It is cobbered by popq %rbx
; CHECK: movq
; CHECK-NEXT: [[LABEL:Ltmp[0-9]*]]
; CHECK: .loc	1 19 0
; CHECK: popq
; CHECK-NEXT: [[CLOBBER:Ltmp[0-9]*]]


; CHECK: Ldebug_loc0:
; CHECK-NEXT: .quad   Lfunc_begin0
; CHECK-NEXT: .quad   [[LABEL]]
; CHECK-NEXT: Lset{{.*}} = Ltmp{{.*}}-Ltmp{{.*}}               ## Loc expr size
; CHECK-NEXT: .short  Lset{{.*}}
; CHECK-NEXT: Ltmp{{.*}}:
; CHECK-NEXT: .byte   85
; CHECK-NEXT: Ltmp{{.*}}:
; CHECK-NEXT: .quad   [[LABEL]]
; CHECK-NEXT: .quad   [[CLOBBER]]
; CHECK-NEXT: Lset{{.*}} = Ltmp{{.*}}-Ltmp{{.*}}               ## Loc expr size
; CHECK-NEXT: .short  Lset{{.*}}
; CHECK-NEXT: Ltmp{{.*}}:
; CHECK-NEXT: .byte   83
; CHECK-NEXT: Ltmp{{.*}}:
!38 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
