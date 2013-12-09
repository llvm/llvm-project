; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Check that the friend tag is there and is followed by a DW_AT_friend that has a reference back.

; CHECK: [[BACK:0x[0-9a-f]*]]:   DW_TAG_class_type
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]       ( .debug_str[{{.*}}] = "A")
; CHECK: DW_TAG_friend
; CHECK-NEXT: DW_AT_friend [DW_FORM_ref4]   (cu + 0x{{[0-9a-f]*}} => {[[BACK]]})


%class.A = type { i32 }
%class.B = type { i32 }

@a = global %class.A zeroinitializer, align 4
@b = global %class.B zeroinitializer, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = metadata !{i32 786449, metadata !28, i32 4, metadata !"clang version 3.1 (trunk 153413) (llvm/trunk 153428)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !17}
!5 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 10, metadata !7, i32 0, i32 1, %class.A* @a, null} ; [ DW_TAG_variable ]
!6 = metadata !{i32 786473, metadata !28} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786434, metadata !28, null, metadata !"A", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!8 = metadata !{metadata !9, metadata !11}
!9 = metadata !{i32 786445, metadata !28, metadata !7, metadata !"a", i32 2, i64 32, i64 32, i64 0, i32 1, metadata !10} ; [ DW_TAG_member ]
!10 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!11 = metadata !{i32 786478, metadata !6, metadata !7, metadata !"A", metadata !"A", metadata !"", i32 1, metadata !12, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !15, i32 1} ; [ DW_TAG_subprogram ]
!12 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !13, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{null, metadata !14}
!14 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !7} ; [ DW_TAG_pointer_type ]
!15 = metadata !{metadata !16}
!16 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!17 = metadata !{i32 786484, i32 0, null, metadata !"b", metadata !"b", metadata !"", metadata !6, i32 11, metadata !18, i32 0, i32 1, %class.B* @b, null} ; [ DW_TAG_variable ]
!18 = metadata !{i32 786434, metadata !28, null, metadata !"B", i32 5, i64 32, i64 32, i32 0, i32 0, null, metadata !19, i32 0, null, null, null} ; [ DW_TAG_class_type ] [B] [line 5, size 32, align 32, offset 0] [def] [from ]
!19 = metadata !{metadata !20, metadata !21, metadata !27}
!20 = metadata !{i32 786445, metadata !28, metadata !18, metadata !"b", i32 7, i64 32, i64 32, i64 0, i32 1, metadata !10} ; [ DW_TAG_member ]
!21 = metadata !{i32 786478, metadata !6, metadata !18, metadata !"B", metadata !"B", metadata !"", i32 5, metadata !22, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !25, i32 5} ; [ DW_TAG_subprogram ]
!22 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !23, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = metadata !{null, metadata !24}
!24 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !18} ; [ DW_TAG_pointer_type ]
!25 = metadata !{metadata !26}
!26 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!27 = metadata !{i32 786474, metadata !18, null, metadata !6, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !7} ; [ DW_TAG_friend ]
!28 = metadata !{metadata !"foo.cpp", metadata !"/Users/echristo/tmp"}
!29 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
