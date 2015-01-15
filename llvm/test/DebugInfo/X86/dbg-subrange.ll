; RUN: llc -O0 < %s | FileCheck %s
; Radar 10464995
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

@s = common global [4294967296 x i8] zeroinitializer, align 16
; CHECK: .quad 4294967296 ## DW_AT_count

define void @bar() nounwind uwtable ssp {
entry:
  store i8 97, i8* getelementptr inbounds ([4294967296 x i8]* @s, i32 0, i64 0), align 1, !dbg !18
  ret void, !dbg !20
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22}

!0 = !{!"0x11\0012\00clang version 3.1 (trunk 144833)\000\00\000\00\000", !21, !1, !1, !3, !11,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00bar\00bar\00\004\000\001\000\006\00256\000\000", !21, !6, !7, null, void ()* @bar, null, null, null} ; [ DW_TAG_subprogram ] [line 4] [def] [scope 0] [bar]
!6 = !{!"0x29", !21} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null}
!11 = !{!13}
!13 = !{!"0x34\00s\00s\00\002\000\001", null, !6, !14, [4294967296 x i8]* @s, null} ; [ DW_TAG_variable ]
!14 = !{!"0x1\00\000\0034359738368\008\000\000", null, null, !15, !16, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 34359738368, align 8, offset 0] [from char]
!15 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!16 = !{!17}
!17 = !{!"0x21\000\004294967296"} ; [ DW_TAG_subrange_type ]
!18 = !MDLocation(line: 5, column: 3, scope: !19)
!19 = !{!"0xb\004\001\000", !21, !5} ; [ DW_TAG_lexical_block ]
!20 = !MDLocation(line: 6, column: 1, scope: !19)
!21 = !{!"small.c", !"/private/tmp"}
!22 = !{i32 1, !"Debug Info Version", i32 2}
