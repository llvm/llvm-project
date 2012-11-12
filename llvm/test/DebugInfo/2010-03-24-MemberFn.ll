; RUN: llc -O0 < %s | grep AT_decl_file |  grep 2
; Here _ZN1S3fooEv is defined in header file identified as AT_decl_file no. 2 in debug info.
%struct.S = type <{ i8 }>

define i32 @_Z3barv() nounwind ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %s1 = alloca %struct.S                          ; <%struct.S*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.S* %s1}, metadata !0), !dbg !16
  %1 = call i32 @_ZN1S3fooEv(%struct.S* %s1) nounwind, !dbg !17 ; <i32> [#uses=1]
  store i32 %1, i32* %0, align 4, !dbg !17
  %2 = load i32* %0, align 4, !dbg !17            ; <i32> [#uses=1]
  store i32 %2, i32* %retval, align 4, !dbg !17
  br label %return, !dbg !17

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !17          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !16
}

define linkonce_odr i32 @_ZN1S3fooEv(%struct.S* %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca %struct.S*                  ; <%struct.S**> [#uses=1]
  %retval = alloca i32                            ; <i32*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{%struct.S** %this_addr}, metadata !18), !dbg !21
  store %struct.S* %this, %struct.S** %this_addr
  br label %return, !dbg !21

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !21          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !22
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!0 = metadata !{i32 524544, metadata !1, metadata !"s1", metadata !4, i32 3, metadata !9} ; [ DW_TAG_auto_variable ]
!1 = metadata !{i32 524299, metadata !2, i32 3, i32 0} ; [ DW_TAG_lexical_block ]
!2 = metadata !{i32 524299, metadata !3, i32 3, i32 0} ; [ DW_TAG_lexical_block ]
!3 = metadata !{i32 524334, i32 0, metadata !4, metadata !"bar", metadata !"bar", metadata !"_Z3barv", metadata !4, i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!4 = metadata !{i32 524329, metadata !"one.cc", metadata !"/tmp/", metadata !5} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 524305, i32 0, i32 4, metadata !"one.cc", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!6 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null} ; [ DW_TAG_subroutine_type ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 524324, metadata !4, metadata !"int", metadata !4, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!9 = metadata !{i32 524307, metadata !4, metadata !"S", metadata !10, i32 2, i64 8, i64 8, i64 0, i32 0, null, metadata !11, i32 0, null} ; [ DW_TAG_structure_type ]
!10 = metadata !{i32 524329, metadata !"one.h", metadata !"/tmp/", metadata !5} ; [ DW_TAG_file_type ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 524334, i32 0, metadata !9, metadata !"foo", metadata !"foo", metadata !"_ZN1S3fooEv", metadata !10, i32 3, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 524309, metadata !4, metadata !"", metadata !4, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !14, i32 0, null} ; [ DW_TAG_subroutine_type ]
!14 = metadata !{metadata !8, metadata !15}
!15 = metadata !{i32 524303, metadata !4, metadata !"", metadata !4, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !9} ; [ DW_TAG_pointer_type ]
!16 = metadata !{i32 3, i32 0, metadata !1, null}
!17 = metadata !{i32 3, i32 0, metadata !3, null}
!18 = metadata !{i32 524545, metadata !12, metadata !"this", metadata !10, i32 3, metadata !19} ; [ DW_TAG_arg_variable ]
!19 = metadata !{i32 524326, metadata !4, metadata !"", metadata !4, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !20} ; [ DW_TAG_const_type ]
!20 = metadata !{i32 524303, metadata !4, metadata !"", metadata !4, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ]
!21 = metadata !{i32 3, i32 0, metadata !12, null}
!22 = metadata !{i32 3, i32 0, metadata !23, null}
!23 = metadata !{i32 524299, metadata !12, i32 3, i32 0} ; [ DW_TAG_lexical_block ]
