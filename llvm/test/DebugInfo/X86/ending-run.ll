; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-dump=line %t | FileCheck %s

; Check that the line table starts at 7, not 4, but that the first
; statement isn't until line 8.

; CHECK-NOT: 0x0000000000000000      7      0      1   0  0  is_stmt
; CHECK: 0x0000000000000000      7      0      1   0
; CHECK: 0x0000000000000004      8     18      1   0  0  is_stmt prologue_end

define i32 @callee(i32 %x) nounwind uwtable ssp {
entry:
  %x.addr = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !12, metadata !{!"0x102"}), !dbg !13
  call void @llvm.dbg.declare(metadata i32* %y, metadata !14, metadata !{!"0x102"}), !dbg !16
  %0 = load i32* %x.addr, align 4, !dbg !17
  %1 = load i32* %x.addr, align 4, !dbg !17
  %mul = mul nsw i32 %0, %1, !dbg !17
  store i32 %mul, i32* %y, align 4, !dbg !17
  %2 = load i32* %y, align 4, !dbg !18
  %sub = sub nsw i32 %2, 2, !dbg !18
  ret i32 %sub, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = !{!"0x11\0012\00clang version 3.1 (trunk 153921) (llvm/trunk 153916)\000\00\000\00\000", !19, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00callee\00callee\00\004\000\001\000\006\000\000\007", !19, !6, !7, null, i32 (i32)* @callee, null, null, null} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !19} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!12 = !{!"0x101\00x\0016777221\000", !5, !6, !9} ; [ DW_TAG_arg_variable ]
!13 = !MDLocation(line: 5, column: 5, scope: !5)
!14 = !{!"0x100\00y\008\000", !15, !6, !9} ; [ DW_TAG_auto_variable ]
!15 = !{!"0xb\007\001\000", !19, !5} ; [ DW_TAG_lexical_block ]
!16 = !MDLocation(line: 8, column: 9, scope: !15)
!17 = !MDLocation(line: 8, column: 18, scope: !15)
!18 = !MDLocation(line: 9, column: 5, scope: !15)
!19 = !{!"ending-run.c", !"/Users/echristo/tmp"}
!20 = !{i32 1, !"Debug Info Version", i32 2}
