; RUN: llc -O2 -asm-verbose < %s | FileCheck %s
; Radar 8616981

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin11.0.0"

%struct.bar = type { i32, i32 }

define i32 @foo(%struct.bar* nocapture %i) nounwind readnone optsize noinline ssp {
; CHECK: TAG_formal_parameter
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.bar* %i}, i64 0, metadata !6), !dbg !12
  ret i32 1, !dbg !13
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!19}

!0 = metadata !{i32 786478, metadata !17, metadata !1, metadata !"foo", metadata !"foo", metadata !"", i32 3, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (%struct.bar*)* @foo, null, null, metadata !16, i32 3} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !17} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !17, i32 12, metadata !"clang version 2.9 (trunk 117922)", i1 true, metadata !"", i32 0, metadata !18, metadata !18, metadata !15, null,  null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !17, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, metadata !17, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 786689, metadata !0, metadata !"i", metadata !1, i32 3, metadata !7, i32 0, null} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 786447, metadata !17, metadata !1, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{i32 786451, metadata !17, metadata !1, metadata !"bar", i32 2, i64 64, i64 32, i64 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [bar] [line 2, size 64, align 32, offset 0] [def] [from ]
!9 = metadata !{metadata !10, metadata !11}
!10 = metadata !{i32 786445, metadata !17,  metadata !1, metadata !"x", i32 2, i64 32, i64 32, i64 0, i32 0, metadata !5} ; [ DW_TAG_member ]
!11 = metadata !{i32 786445, metadata !17, metadata !1, metadata !"y", i32 2, i64 32, i64 32, i64 32, i32 0, metadata !5} ; [ DW_TAG_member ]
!12 = metadata !{i32 3, i32 47, metadata !0, null}
!13 = metadata !{i32 4, i32 2, metadata !14, null}
!14 = metadata !{i32 786443, metadata !17, metadata !0, i32 3, i32 50, i32 0} ; [ DW_TAG_lexical_block ]
!15 = metadata !{metadata !0}
!16 = metadata !{metadata !6}
!17 = metadata !{metadata !"one.c", metadata !"/private/tmp"}
!18 = metadata !{i32 0}
!19 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
