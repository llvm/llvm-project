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
  call void @llvm.dbg.declare(metadata [2 x i32]* %i, metadata !10, metadata !{!"0x102"}), !dbg !15
  ret i32 0, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = !{!"0x11\0012\00clang version 3.3 (trunk 171472) (llvm/trunk 171487)\000\00\000\00\000", !17, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.c] [DW_LANG_C99]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00main\00main\00\002\000\001\000\006\00256\000\003", !6, !6, !7, null, i32 ()* @main, null, null, !1} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 3] [main]
!6 = !{!"0x29", !17} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x100\00i\004\000", !11, !6, !12} ; [ DW_TAG_auto_variable ] [i] [line 4]
!11 = !{!"0xb\003\000\000", !6, !5} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/foo.c]
!12 = !{!"0x1\00\000\0064\0032\000\000", null, null, !9, !13, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 64, align 32, offset 0] [from int]
!13 = !{!14}
!14 = !{!"0x21\000\002"}        ; [ DW_TAG_subrange_type ] [0, 1]
!15 = !MDLocation(line: 4, scope: !11)
!16 = !MDLocation(line: 6, scope: !11)
!17 = !{!"foo.c", !"/usr/local/google/home/echristo/tmp"}
!18 = !{i32 1, !"Debug Info Version", i32 2}
