; RUN: llc -O0 -asm-verbose -mtriple=x86_64-macosx -generate-dwarf-pub-sections=Enable < %s | FileCheck %s
; CHECK-NOT: .asciz "X" ## External Name
; CHECK: .asciz "Y" ## External Name
; Test to check type with no definition is listed in pubtypes section.
%struct.X = type opaque
%struct.Y = type { i32 }

define i32 @foo(%struct.X* %x, %struct.Y* %y) nounwind ssp {
entry:
  %x_addr = alloca %struct.X*                     ; <%struct.X**> [#uses=1]
  %y_addr = alloca %struct.Y*                     ; <%struct.Y**> [#uses=1]
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata %struct.X** %x_addr, metadata !0, metadata !{!"0x102"}), !dbg !13
  store %struct.X* %x, %struct.X** %x_addr
  call void @llvm.dbg.declare(metadata %struct.Y** %y_addr, metadata !14, metadata !{!"0x102"}), !dbg !13
  store %struct.Y* %y, %struct.Y** %y_addr
  store i32 0, i32* %0, align 4, !dbg !13
  %1 = load i32* %0, align 4, !dbg !13            ; <i32> [#uses=1]
  store i32 %1, i32* %retval, align 4, !dbg !13
  br label %return, !dbg !13

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !13          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!20}

!0 = !{!"0x101\00x\007\000", !1, !2, !7} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00foo\00foo\00foo\007\000\001\000\006\000\000\007", !18, !2, !4, null, i32 (%struct.X*, %struct.Y*)* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !18} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\000\00\000\00\000", !18, !19, !19, !17, null,  null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !18, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!6, !7, !9}
!6 = !{!"0x24\00int\000\0032\0032\000\000\005", !18, !2} ; [ DW_TAG_base_type ]
!7 = !{!"0xf\00\000\0064\0064\000\000", !18, !2, !8} ; [ DW_TAG_pointer_type ]
!8 = !{!"0x13\00X\003\000\000\000\004\000", !18, !2, null, null, null, null, null} ; [ DW_TAG_structure_type ] [X] [line 3, size 0, align 0, offset 0] [decl] [from ]
!9 = !{!"0xf\00\000\0064\0064\000\000", !18, !2, !10} ; [ DW_TAG_pointer_type ]
!10 = !{!"0x13\00Y\004\0032\0032\000\000\000", !18, !2, null, !11, null, null, null} ; [ DW_TAG_structure_type ] [Y] [line 4, size 32, align 32, offset 0] [def] [from ]
!11 = !{!12}
!12 = !{!"0xd\00x\005\0032\0032\000\000", !18, !10, !6} ; [ DW_TAG_member ]
!13 = !MDLocation(line: 7, scope: !1)
!14 = !{!"0x101\00y\007\000", !1, !2, !9} ; [ DW_TAG_arg_variable ]
!15 = !MDLocation(line: 7, scope: !16)
!16 = !{!"0xb\007\000\000", !18, !1} ; [ DW_TAG_lexical_block ]
!17 = !{!1}
!18 = !{!"a.c", !"/tmp/"}
!19 = !{i32 0}
!20 = !{i32 1, !"Debug Info Version", i32 2}
