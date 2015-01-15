; RUN: llvm-link %s %p/2011-08-04-Metadata2.ll -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s
; Test if internal global variable's debug info is merged appropriately or not.

;CHECK:   !"0x34\00x\00x\00\002\001\001", !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, i32* @x}
;CHECK:   !"0x34\00x\00x\00\001\001\001", !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, i32* @x1}
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

@x = internal global i32 0, align 4

define void @foo() nounwind uwtable ssp {
entry:
  store i32 1, i32* @x, align 4, !dbg !7
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}
!llvm.dbg.sp = !{!1}
!llvm.dbg.gv = !{!5}

!0 = !{!"0x11\0012\00clang version 3.0 ()\001\00\000\00\000", !9, !4, !4, !10, null, null} ; [ DW_TAG_compile_unit ]
!1 = !{!"0x2e\00foo\00foo\00\003\000\001\000\006\000\000\000", !9, !2, !3, null, void ()* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 0] [foo]
!2 = !{!"0x29", !9} ; [ DW_TAG_file_type ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !9, !2, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null}
!5 = !{!"0x34\00x\00x\00\002\001\001", !0, !2, !6, i32* @x} ; [ DW_TAG_variable ]
!6 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !0} ; [ DW_TAG_base_type ]
!7 = !MDLocation(line: 3, column: 14, scope: !8)
!8 = !{!"0xb\003\0012\000", !9, !1} ; [ DW_TAG_lexical_block ]
!9 = !{!"/tmp/one.c", !"/Volumes/Lalgate/Slate/D"}
!10 = !{!1}
!11 = !{i32 1, !"Debug Info Version", i32 2}
