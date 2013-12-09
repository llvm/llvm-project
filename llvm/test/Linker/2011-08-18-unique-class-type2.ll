; This file is for use with 2011-08-10-unique-class-type.ll
; RUN: true

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

%"class.N1::A" = type { i8 }

define void @_Z3barN2N11AE() nounwind uwtable ssp {
entry:
  %youra = alloca %"class.N1::A", align 1
  call void @llvm.dbg.declare(metadata !{%"class.N1::A"* %youra}, metadata !9), !dbg !13
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = metadata !{i32 720913, metadata !16, i32 4, metadata !"clang version 3.0 (trunk 137954)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 720942, i32 0, metadata !6, metadata !"bar", metadata !"bar", metadata !"_Z3barN2N11AE", i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z3barN2N11AE, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 4] [def] [scope 0] [bar]
!6 = metadata !{i32 720937, metadata !16} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, metadata !16, metadata !6, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{i32 721153, metadata !5, metadata !"youra", metadata !6, i32 16777220, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!10 = metadata !{i32 720898, metadata !17, metadata !11, metadata !"A", i32 3, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_class_type ] [A] [line 3, size 8, align 8, offset 0] [def] [from ]
!11 = metadata !{i32 720953, metadata !17, null, metadata !"N1", i32 2} ; [ DW_TAG_namespace ]
!12 = metadata !{i32 720937, metadata !17} ; [ DW_TAG_file_type ]
!13 = metadata !{i32 4, i32 12, metadata !5, null}
!14 = metadata !{i32 4, i32 20, metadata !15, null}
!15 = metadata !{i32 720907, metadata !16, metadata !5, i32 4, i32 19, i32 0} ; [ DW_TAG_lexical_block ]
!16 = metadata !{metadata !"n2.c", metadata !"/private/tmp"}
!17 = metadata !{metadata !"./n.h", metadata !"/private/tmp"}
!18 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
