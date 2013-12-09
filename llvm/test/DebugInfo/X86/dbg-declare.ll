; RUN: llc < %s -O0 -mtriple x86_64-apple-darwin
; <rdar://problem/11134152>

define i32 @foo(i32* %x) nounwind uwtable ssp {
entry:
  %x.addr = alloca i32*, align 8
  %saved_stack = alloca i8*
  %cleanup.dest.slot = alloca i32
  store i32* %x, i32** %x.addr, align 8
  call void @llvm.dbg.declare(metadata !{i32** %x.addr}, metadata !14), !dbg !15
  %0 = load i32** %x.addr, align 8, !dbg !16
  %1 = load i32* %0, align 4, !dbg !16
  %2 = zext i32 %1 to i64, !dbg !16
  %3 = call i8* @llvm.stacksave(), !dbg !16
  store i8* %3, i8** %saved_stack, !dbg !16
  %vla = alloca i8, i64 %2, align 16, !dbg !16
  call void @llvm.dbg.declare(metadata !{i8* %vla}, metadata !18), !dbg !23
  store i32 1, i32* %cleanup.dest.slot
  %4 = load i8** %saved_stack, !dbg !24
  call void @llvm.stackrestore(i8* %4), !dbg !24
  ret i32 0, !dbg !25
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i8* @llvm.stacksave() nounwind

declare void @llvm.stackrestore(i8*) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = metadata !{i32 786449, metadata !26, i32 12, metadata !"clang version 3.1 (trunk 153698)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !26, metadata !0, metadata !"foo", metadata !"foo", metadata !"", i32 6, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32*)* @foo, null, null, metadata !12, i32 0} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !26} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 786447, null, null, null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 786470, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_const_type ]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!14 = metadata !{i32 786689, metadata !5, metadata !"x", metadata !6, i32 16777221, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 5, i32 21, metadata !5, null}
!16 = metadata !{i32 7, i32 13, metadata !17, null}
!17 = metadata !{i32 786443, metadata !26, metadata !5, i32 6, i32 1, i32 0} ; [ DW_TAG_lexical_block ]
!18 = metadata !{i32 786688, metadata !17, metadata !"a", metadata !6, i32 7, metadata !19, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!19 = metadata !{i32 786433, null, null, null, i32 0, i64 0, i64 8, i32 0, i32 0, metadata !20, metadata !21, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 8, offset 0] [from char]
!20 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!21 = metadata !{metadata !22}
!22 = metadata !{i32 786465, i64 0, i64 -1}        ; [ DW_TAG_subrange_type ]
!23 = metadata !{i32 7, i32 8, metadata !17, null}
!24 = metadata !{i32 9, i32 1, metadata !17, null}
!25 = metadata !{i32 8, i32 3, metadata !17, null}
!26 = metadata !{metadata !"20020104-2.c", metadata !"/Volumes/Sandbox/llvm"}
!27 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
