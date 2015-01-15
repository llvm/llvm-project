; RUN: opt -mem2reg < %s | llvm-dis | grep ".dbg " | count 7

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @foo(i32, i64, i8*)

define void @baz(i32 %a) nounwind ssp {
entry:
  %x_addr.i = alloca i32                          ; <i32*> [#uses=2]
  %y_addr.i = alloca i64                          ; <i64*> [#uses=2]
  %z_addr.i = alloca i8*                          ; <i8**> [#uses=2]
  %a_addr = alloca i32                            ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata i32* %a_addr, metadata !0, metadata !{}), !dbg !7
  store i32 %a, i32* %a_addr
  %0 = load i32* %a_addr, align 4, !dbg !8        ; <i32> [#uses=1]
  call void @llvm.dbg.declare(metadata i32* %x_addr.i, metadata !9, metadata !{}) nounwind, !dbg !15
  store i32 %0, i32* %x_addr.i
  call void @llvm.dbg.declare(metadata i64* %y_addr.i, metadata !16, metadata !{}) nounwind, !dbg !15
  store i64 55, i64* %y_addr.i
  call void @llvm.dbg.declare(metadata i8** %z_addr.i, metadata !17, metadata !{}) nounwind, !dbg !15
  store i8* bitcast (void (i32)* @baz to i8*), i8** %z_addr.i
  %1 = load i32* %x_addr.i, align 4, !dbg !18     ; <i32> [#uses=1]
  %2 = load i64* %y_addr.i, align 8, !dbg !18     ; <i64> [#uses=1]
  %3 = load i8** %z_addr.i, align 8, !dbg !18     ; <i8*> [#uses=1]
  call void @foo(i32 %1, i64 %2, i8* %3) nounwind, !dbg !18
  br label %return, !dbg !19

return:                                           ; preds = %entry
  ret void, !dbg !19
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!22}
!0 = !{!"0x101\00a\008\000", !1, !2, !6} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00baz\00baz\00baz\008\000\001\000\006\000\000\008", !20, !2, !4, null, void (i32)* @baz, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", !20, !21, !21, null, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !20, !2, null, !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{null, !6}
!6 = !{!"0x24\00int\000\0032\0032\000\000\005", !20, !2} ; [ DW_TAG_base_type ]
!7 = !MDLocation(line: 8, scope: !1)
!8 = !MDLocation(line: 9, scope: !1)
!9 = !{!"0x101\00x\004\000", !10, !2, !6} ; [ DW_TAG_arg_variable ]
!10 = !{!"0x2e\00bar\00bar\00bar\004\001\001\000\006\000\000\004", !20, !2, !11, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!11 = !{!"0x15\00\000\000\000\000\000\000", !20, !2, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{null, !6, !13, !14}
!13 = !{!"0x24\00long int\000\0064\0064\000\000\005", !20, !2} ; [ DW_TAG_base_type ]
!14 = !{!"0xf\00\000\0064\0064\000\000", !20, !2, null} ; [ DW_TAG_pointer_type ]
!15 = !MDLocation(line: 4, scope: !10, inlinedAt: !8)
!16 = !{!"0x101\00y\004\000", !10, !2, !13} ; [ DW_TAG_arg_variable ]
!17 = !{!"0x101\00z\004\000", !10, !2, !14} ; [ DW_TAG_arg_variable ]
!18 = !MDLocation(line: 5, scope: !10, inlinedAt: !8)
!19 = !MDLocation(line: 10, scope: !1)
!20 = !{!"bar.c", !"/tmp/"}
!21 = !{i32 0}
!22 = !{i32 1, !"Debug Info Version", i32 2}
