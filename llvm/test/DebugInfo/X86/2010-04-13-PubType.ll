; RUN: llc -O0 -asm-verbose -mtriple=x86_64-macosx < %s | FileCheck %s
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
  call void @llvm.dbg.declare(metadata !{%struct.X** %x_addr}, metadata !0), !dbg !13
  store %struct.X* %x, %struct.X** %x_addr
  call void @llvm.dbg.declare(metadata !{%struct.Y** %y_addr}, metadata !14), !dbg !13
  store %struct.Y* %y, %struct.Y** %y_addr
  store i32 0, i32* %0, align 4, !dbg !13
  %1 = load i32* %0, align 4, !dbg !13            ; <i32> [#uses=1]
  store i32 %1, i32* %retval, align 4, !dbg !13
  br label %return, !dbg !13

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !13          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!0 = metadata !{i32 524545, metadata !1, metadata !"x", metadata !2, i32 7, metadata !7} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 524334, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"foo", metadata !2, i32 7, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 524329, metadata !"a.c", metadata !"/tmp/", metadata !3} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 524305, i32 0, i32 1, metadata !"a.c", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{metadata !6, metadata !7, metadata !9}
!6 = metadata !{i32 524324, metadata !2, metadata !"int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 524303, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{i32 524307, metadata !2, metadata !"X", metadata !2, i32 3, i64 0, i64 0, i64 0, i32 4, null, null, i32 0, null} ; [ DW_TAG_structure_type ]
!9 = metadata !{i32 524303, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 524307, metadata !2, metadata !"Y", metadata !2, i32 4, i64 32, i64 32, i64 0, i32 0, null, metadata !11, i32 0, null} ; [ DW_TAG_structure_type ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 524301, metadata !10, metadata !"x", metadata !2, i32 5, i64 32, i64 32, i64 0, i32 0, metadata !6} ; [ DW_TAG_member ]
!13 = metadata !{i32 7, i32 0, metadata !1, null}
!14 = metadata !{i32 524545, metadata !1, metadata !"y", metadata !2, i32 7, metadata !9} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 7, i32 0, metadata !16, null}
!16 = metadata !{i32 524299, metadata !1, i32 7, i32 0} ; [ DW_TAG_lexical_block ]
