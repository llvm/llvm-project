; RUN: llc -O0 -relocation-model pic < %s -o /dev/null
; PR7545
@.str = private constant [4 x i8] c"one\00", align 1 ; <[4 x i8]*> [#uses=1]
@.str1 = private constant [4 x i8] c"two\00", align 1 ; <[5 x i8]*> [#uses=1]
@C.9.2167 = internal constant [2 x i8*] [i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8]* @.str1, i64 0, i64 0)]
!38 = !{!"0x29", !109} ; [ DW_TAG_file_type ]
!39 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)\001\00\000\00\000", !109, !108, !108, null, null, null} ; [ DW_TAG_compile_unit ]
!46 = !{!"0xf\00\000\0064\0064\000\000", !109, null, !47} ; [ DW_TAG_pointer_type ]
!47 = !{!"0x24\00char\000\008\008\000\000\006", !109, null} ; [ DW_TAG_base_type ]
!97 = !{!"0x2e\00main\00main\00main\0073\000\001\000\006\000\000\000", i32 0, !39, !98, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!98 = !{!"0x15\00\000\000\000\000\000\000", !109, null, null, !99, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!99 = !{!100}
!100 = !{!"0x24\00int\000\0032\0032\000\000\005", !109, null} ; [ DW_TAG_base_type ]
!101 = !{[2 x i8*]* @C.9.2167}
!102 = !{!"0x100\00find_strings\0075\000", !103, !38, !104} ; [ DW_TAG_auto_variable ]
!103 = !{!"0xb\0073\000\000", null, !97} ; [ DW_TAG_lexical_block ]
!104 = !{!"0x1\00\000\0085312\0064\000\000", !109, null, !46, !105, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 85312, align 64, offset 0] [from ]
!105 = !{!106}
!106 = !{!"0x21\000\001333"}    ; [ DW_TAG_subrange_type ]
!107 = !MDLocation(line: 73, scope: !103)
!108 = !{i32 0}
!109 = !{!"pbmsrch.c", !"/Users/grawp/LLVM/test-suite/MultiSource/Benchmarks/MiBench/office-stringsearch"}

define i32 @main() nounwind ssp {
bb.nph:
  tail call void @llvm.dbg.declare(metadata [2 x i8*]* @C.9.2167, metadata !102, metadata !{!"0x102"}), !dbg !107
  ret i32 0, !dbg !107
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

