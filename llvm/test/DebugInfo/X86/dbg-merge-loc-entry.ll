; RUN: llc < %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; RUN: llc < %s -o %t -filetype=obj -regalloc=basic
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin8"

;CHECK: DW_AT_location{{.*}}(<0x1> 55 )

%0 = type { i64, i1 }

@__clz_tab = external constant [256 x i8]

define hidden i128 @__divti3(i128 %u, i128 %v) nounwind readnone {
entry:
  tail call void @llvm.dbg.value(metadata i128 %u, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !15
  tail call void @llvm.dbg.value(metadata i64 0, i64 0, metadata !17, metadata !{!"0x102"}), !dbg !21
  br i1 undef, label %bb2, label %bb4, !dbg !22

bb2:                                              ; preds = %entry
  br label %bb4, !dbg !23

bb4:                                              ; preds = %bb2, %entry
  br i1 undef, label %__udivmodti4.exit, label %bb82.i, !dbg !24

bb82.i:                                           ; preds = %bb4
  unreachable

__udivmodti4.exit:                                ; preds = %bb4
  ret i128 undef, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

declare %0 @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!32}

!0 = !{!"0x2e\00__udivmodti4\00__udivmodti4\00\00879\001\001\000\006\00256\001\00879", !29, !1, !3, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !29} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", !29, !31, !31, !28, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !29, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5, !5, !5, !8}
!5 = !{!"0x16\00UTItype\00166\000\000\000\000", !30, !6, !7} ; [ DW_TAG_typedef ]
!6 = !{!"0x29", !30} ; [ DW_TAG_file_type ]
!7 = !{!"0x24\00\000\00128\00128\000\000\007", !29, !1} ; [ DW_TAG_base_type ]
!8 = !{!"0xf\00\000\0064\0064\000\000", !29, !1, !5} ; [ DW_TAG_pointer_type ]
!9 = !{!"0x2e\00__divti3\00__divti3\00__divti3\001094\000\001\000\006\00256\001\001094", !29, !1, !10, null, i128 (i128, i128)* @__divti3, null, null, null} ; [ DW_TAG_subprogram ]
!10 = !{!"0x15\00\000\000\000\000\000\000", !29, !1, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{!12, !12, !12}
!12 = !{!"0x16\00TItype\00160\000\000\000\000", !30, !6, !13} ; [ DW_TAG_typedef ]
!13 = !{!"0x24\00\000\00128\00128\000\000\005", !29, !1} ; [ DW_TAG_base_type ]
!14 = !{!"0x101\00u\001093\000", !9, !1, !12} ; [ DW_TAG_arg_variable ]
!15 = !MDLocation(line: 1093, scope: !9)
!16 = !{i64 0}
!17 = !{!"0x100\00c\001095\000", !18, !1, !19} ; [ DW_TAG_auto_variable ]
!18 = !{!"0xb\001094\000\0013", !29, !9} ; [ DW_TAG_lexical_block ]
!19 = !{!"0x16\00word_type\00424\000\000\000\000", !30, !6, !20} ; [ DW_TAG_typedef ]
!20 = !{!"0x24\00long int\000\0064\0064\000\000\005", !29, !1} ; [ DW_TAG_base_type ]
!21 = !MDLocation(line: 1095, scope: !18)
!22 = !MDLocation(line: 1103, scope: !18)
!23 = !MDLocation(line: 1104, scope: !18)
!24 = !MDLocation(line: 1003, scope: !25, inlinedAt: !26)
!25 = !{!"0xb\00879\000\000", !29, !0} ; [ DW_TAG_lexical_block ]
!26 = !MDLocation(line: 1107, scope: !18)
!27 = !MDLocation(line: 1111, scope: !18)
!28 = !{!0, !9}
!29 = !{!"foobar.c", !"/tmp"}
!30 = !{!"foobar.h", !"/tmp"}
!31 = !{i32 0}
!32 = !{i32 1, !"Debug Info Version", i32 2}
