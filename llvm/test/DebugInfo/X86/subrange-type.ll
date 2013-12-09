; RUN: llc -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Make sure that the base type from the subrange type has a name.
; CHECK: DW_TAG_subrange_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]     (cu + 0x{{[0-9a-f]+}} => {[[SUBTYPE:0x[0-9a-f]*]]})
; CHECK: [[SUBTYPE]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %i = alloca [2 x i32], align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata !{[2 x i32]* %i}, metadata !10), !dbg !15
  ret i32 0, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = metadata !{i32 786449, metadata !17, i32 12, metadata !"clang version 3.3 (trunk 171472) (llvm/trunk 171487)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.c] [DW_LANG_C99]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !6, metadata !6, metadata !"main", metadata !"main", metadata !"", i32 2, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !1, i32 3} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 3] [main]
!6 = metadata !{i32 786473, metadata !17} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786688, metadata !11, metadata !"i", metadata !6, i32 4, metadata !12, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [i] [line 4]
!11 = metadata !{i32 786443, metadata !6, metadata !5, i32 3, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/foo.c]
!12 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 64, i64 32, i32 0, i32 0, metadata !9, metadata !13, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 64, align 32, offset 0] [from int]
!13 = metadata !{metadata !14}
!14 = metadata !{i32 786465, i64 0, i64 2}        ; [ DW_TAG_subrange_type ] [0, 1]
!15 = metadata !{i32 4, i32 0, metadata !11, null}
!16 = metadata !{i32 6, i32 0, metadata !11, null}
!17 = metadata !{metadata !"foo.c", metadata !"/usr/local/google/home/echristo/tmp"}
!18 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
