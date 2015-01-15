; RUN: llc < %s -O0 -mtriple x86_64-apple-darwin
; <rdar://problem/11134152>

define i32 @foo(i32* %x) nounwind uwtable ssp {
entry:
  %x.addr = alloca i32*, align 8
  %saved_stack = alloca i8*
  %cleanup.dest.slot = alloca i32
  store i32* %x, i32** %x.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %x.addr, metadata !14, metadata !{!"0x102"}), !dbg !15
  %0 = load i32** %x.addr, align 8, !dbg !16
  %1 = load i32* %0, align 4, !dbg !16
  %2 = zext i32 %1 to i64, !dbg !16
  %3 = call i8* @llvm.stacksave(), !dbg !16
  store i8* %3, i8** %saved_stack, !dbg !16
  %vla = alloca i8, i64 %2, align 16, !dbg !16
  call void @llvm.dbg.declare(metadata i8* %vla, metadata !18, metadata !{!"0x102"}), !dbg !23
  store i32 1, i32* %cleanup.dest.slot
  %4 = load i8** %saved_stack, !dbg !24
  call void @llvm.stackrestore(i8* %4), !dbg !24
  ret i32 0, !dbg !25
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i8* @llvm.stacksave() nounwind

declare void @llvm.stackrestore(i8*) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = !{!"0x11\0012\00clang version 3.1 (trunk 153698)\000\00\000\00\000", !26, !1, !1, !3, !1, null} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00foo\00foo\00\006\000\001\000\006\00256\000\000", !26, !0, !7, null, i32 (i32*)* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !26} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !10}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ]
!11 = !{!"0x26\00\000\000\000\000\000", null, null, !9} ; [ DW_TAG_const_type ]
!14 = !{!"0x101\00x\0016777221\000", !5, !6, !10} ; [ DW_TAG_arg_variable ]
!15 = !MDLocation(line: 5, column: 21, scope: !5)
!16 = !MDLocation(line: 7, column: 13, scope: !17)
!17 = !{!"0xb\006\001\000", !26, !5} ; [ DW_TAG_lexical_block ]
!18 = !{!"0x100\00a\007\000", !17, !6, !19} ; [ DW_TAG_auto_variable ]
!19 = !{!"0x1\00\000\000\008\000\000", null, null, !20, !21, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 8, offset 0] [from char]
!20 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!21 = !{!22}
!22 = !{!"0x21\000\00-1"}        ; [ DW_TAG_subrange_type ]
!23 = !MDLocation(line: 7, column: 8, scope: !17)
!24 = !MDLocation(line: 9, column: 1, scope: !17)
!25 = !MDLocation(line: 8, column: 3, scope: !17)
!26 = !{!"20020104-2.c", !"/Volumes/Sandbox/llvm"}
!27 = !{i32 1, !"Debug Info Version", i32 2}
