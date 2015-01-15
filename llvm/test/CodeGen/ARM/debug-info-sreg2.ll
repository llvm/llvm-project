; RUN: llc < %s - -filetype=obj | llvm-dwarfdump -debug-dump=loc - | FileCheck %s
; Radar 9376013
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

; Just making sure the first part of the location isn't a repetition
; of the size of the location description.
;
; 0x90   DW_OP_regx of super-register

; CHECK: 0x00000000: Beginning address offset:
; CHECK-NEXT:           Ending address offset:
; CHECK-NEXT:            Location description: 90 {{.. .. .. .. $}}

define void @_Z3foov() optsize ssp {
entry:
  %call = tail call float @_Z3barv() optsize, !dbg !11
  tail call void @llvm.dbg.value(metadata float %call, i64 0, metadata !5, metadata !{!"0x102"}), !dbg !11
  %call16 = tail call float @_Z2f2v() optsize, !dbg !12
  %cmp7 = fcmp olt float %call, %call16, !dbg !12
  br i1 %cmp7, label %for.body, label %for.end, !dbg !12

for.body:                                         ; preds = %entry, %for.body
  %k.08 = phi float [ %inc, %for.body ], [ %call, %entry ]
  %call4 = tail call float @_Z2f3f(float %k.08) optsize, !dbg !13
  %inc = fadd float %k.08, 1.000000e+00, !dbg !14
  %call1 = tail call float @_Z2f2v() optsize, !dbg !12
  %cmp = fcmp olt float %inc, %call1, !dbg !12
  br i1 %cmp, label %for.body, label %for.end, !dbg !12

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !15
}

declare float @_Z3barv() optsize

declare float @_Z2f2v() optsize

declare float @_Z2f3f(float) optsize

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = !{!"0x11\004\00clang version 3.0 (trunk 130845)\001\00\000\00\001", !18, !19, !19, !16, null,  null} ; [ DW_TAG_compile_unit ]
!1 = !{!"0x2e\00foo\00foo\00_Z3foov\005\000\001\000\006\00256\001\005", !18, !2, !3, null, void ()* @_Z3foov, null, null, !17} ; [ DW_TAG_subprogram ] [line 5] [def] [foo]
!2 = !{!"0x29", !18} ; [ DW_TAG_file_type ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !18, !2, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null}
!5 = !{!"0x100\00k\006\000", !6, !2, !7} ; [ DW_TAG_auto_variable ]
!6 = !{!"0xb\005\0012\000", !18, !1} ; [ DW_TAG_lexical_block ]
!7 = !{!"0x24\00float\000\0032\0032\000\000\004", null, !0} ; [ DW_TAG_base_type ]
!8 = !{!"0x100\00y\008\000", !9, !2, !7} ; [ DW_TAG_auto_variable ]
!9 = !{!"0xb\007\0025\002", !18, !10} ; [ DW_TAG_lexical_block ]
!10 = !{!"0xb\007\003\001", !18, !6} ; [ DW_TAG_lexical_block ]
!11 = !MDLocation(line: 6, column: 18, scope: !6)
!12 = !MDLocation(line: 7, column: 3, scope: !6)
!13 = !MDLocation(line: 8, column: 20, scope: !9)
!14 = !MDLocation(line: 7, column: 20, scope: !10)
!15 = !MDLocation(line: 10, column: 1, scope: !6)
!16 = !{!1}
!17 = !{!5, !8}
!18 = !{!"k.cc", !"/private/tmp"}
!19 = !{i32 0}
!20 = !{i32 1, !"Debug Info Version", i32 2}
