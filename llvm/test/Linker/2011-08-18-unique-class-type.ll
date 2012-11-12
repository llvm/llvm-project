; RUN: llvm-link %s %p/2011-08-18-unique-class-type2.ll -S -o - | grep DW_TAG_class_type | count 1
; Test to check there is only one MDNode for class A after linking.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

%"class.N1::A" = type { i8 }

define void @_Z3fooN2N11AE() nounwind uwtable ssp {
entry:
  %mya = alloca %"class.N1::A", align 1
  call void @llvm.dbg.declare(metadata !{%"class.N1::A"* %mya}, metadata !9), !dbg !13
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 4, metadata !"n1.c", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 137954)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720942, i32 0, metadata !6, metadata !"foo", metadata !"foo", metadata !"_Z3fooN2N11AE", metadata !6, i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void ()* @_Z3fooN2N11AE, null, null} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"n1.c", metadata !"/private/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, metadata !6, metadata !"", metadata !6, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null}
!9 = metadata !{i32 721153, metadata !5, metadata !"mya", metadata !6, i32 16777220, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!10 = metadata !{i32 720898, metadata !11, metadata !"A", metadata !12, i32 3, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null} ; [ DW_TAG_class_type ]
!11 = metadata !{i32 720953, null, metadata !"N1", metadata !12, i32 2} ; [ DW_TAG_namespace ]
!12 = metadata !{i32 720937, metadata !"./n.h", metadata !"/private/tmp", null} ; [ DW_TAG_file_type ]
!13 = metadata !{i32 4, i32 12, metadata !5, null}
!14 = metadata !{i32 4, i32 18, metadata !15, null}
!15 = metadata !{i32 720907, metadata !5, i32 4, i32 17, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
