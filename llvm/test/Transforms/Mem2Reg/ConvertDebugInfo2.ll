; RUN: opt -mem2reg < %s | llvm-dis | grep ".dbg " | count 7

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @foo(i32, i64, i8*)

define void @baz(i32 %a) nounwind ssp {
entry:
  %x_addr.i = alloca i32                          ; <i32*> [#uses=2]
  %y_addr.i = alloca i64                          ; <i64*> [#uses=2]
  %z_addr.i = alloca i8*                          ; <i8**> [#uses=2]
  %a_addr = alloca i32                            ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{i32* %a_addr}, metadata !0), !dbg !7
  store i32 %a, i32* %a_addr
  %0 = load i32* %a_addr, align 4, !dbg !8        ; <i32> [#uses=1]
  call void @llvm.dbg.declare(metadata !{i32* %x_addr.i}, metadata !9) nounwind, !dbg !15
  store i32 %0, i32* %x_addr.i
  call void @llvm.dbg.declare(metadata !{i64* %y_addr.i}, metadata !16) nounwind, !dbg !15
  store i64 55, i64* %y_addr.i
  call void @llvm.dbg.declare(metadata !{i8** %z_addr.i}, metadata !17) nounwind, !dbg !15
  store i8* bitcast (void (i32)* @baz to i8*), i8** %z_addr.i
  %1 = load i32* %x_addr.i, align 4, !dbg !18     ; <i32> [#uses=1]
  %2 = load i64* %y_addr.i, align 8, !dbg !18     ; <i64> [#uses=1]
  %3 = load i8** %z_addr.i, align 8, !dbg !18     ; <i8*> [#uses=1]
  call void @foo(i32 %1, i64 %2, i8* %3) nounwind, !dbg !18
  br label %return, !dbg !19

return:                                           ; preds = %entry
  ret void, !dbg !19
}

!0 = metadata !{i32 524545, metadata !1, metadata !"a", metadata !2, i32 8, metadata !6} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 524334, i32 0, metadata !2, metadata !"baz", metadata !"baz", metadata !"baz", metadata !2, i32 8, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 524329, metadata !"bar.c", metadata !"/tmp/", metadata !3} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 524305, i32 0, i32 1, metadata !"bar.c", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{null, metadata !6}
!6 = metadata !{i32 524324, metadata !2, metadata !"int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 8, i32 0, metadata !1, null}
!8 = metadata !{i32 9, i32 0, metadata !1, null}
!9 = metadata !{i32 524545, metadata !10, metadata !"x", metadata !2, i32 4, metadata !6} ; [ DW_TAG_arg_variable ]
!10 = metadata !{i32 524334, i32 0, metadata !2, metadata !"bar", metadata !"bar", metadata !"bar", metadata !2, i32 4, metadata !11, i1 true, i1 true, i32 0, i32 0, null, i1 false, i1 false} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null} ; [ DW_TAG_subroutine_type ]
!12 = metadata !{null, metadata !6, metadata !13, metadata !14}
!13 = metadata !{i32 524324, metadata !2, metadata !"long int", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 524303, metadata !2, metadata !"", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!15 = metadata !{i32 4, i32 0, metadata !10, metadata !8}
!16 = metadata !{i32 524545, metadata !10, metadata !"y", metadata !2, i32 4, metadata !13} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 524545, metadata !10, metadata !"z", metadata !2, i32 4, metadata !14} ; [ DW_TAG_arg_variable ]
!18 = metadata !{i32 5, i32 0, metadata !10, metadata !8}
!19 = metadata !{i32 10, i32 0, metadata !1, null}
