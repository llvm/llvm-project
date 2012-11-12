; RUN: llc -O0 < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"

;CHECK: .loc	1 2 11 prologue_end
define i32 @foo(i32 %i) nounwind ssp {
entry:
  %i.addr = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !7), !dbg !8
  call void @llvm.dbg.declare(metadata !{i32* %j}, metadata !9), !dbg !11
  store i32 2, i32* %j, align 4, !dbg !12
  %tmp = load i32* %j, align 4, !dbg !13
  %inc = add nsw i32 %tmp, 1, !dbg !13
  store i32 %inc, i32* %j, align 4, !dbg !13
  %tmp1 = load i32* %j, align 4, !dbg !14
  %tmp2 = load i32* %i.addr, align 4, !dbg !14
  %add = add nsw i32 %tmp1, %tmp2, !dbg !14
  store i32 %add, i32* %j, align 4, !dbg !14
  %tmp3 = load i32* %j, align 4, !dbg !15
  ret i32 %tmp3, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 @foo(i32 21), !dbg !16
  ret i32 %call, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1, !6}

!0 = metadata !{i32 589841, i32 0, i32 12, metadata !"/tmp/a.c", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 131100)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 589870, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"", metadata !2, i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, i32 (i32)* @foo, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 589865, metadata !"/tmp/a.c", metadata !"/private/tmp", metadata !0} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 589845, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !0, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 589870, i32 0, metadata !2, metadata !"main", metadata !"main", metadata !"", metadata !2, i32 7, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, i32 ()* @main, null, null} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 590081, metadata !1, metadata !"i", metadata !2, i32 16777217, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!8 = metadata !{i32 1, i32 13, metadata !1, null}
!9 = metadata !{i32 590080, metadata !10, metadata !"j", metadata !2, i32 2, metadata !5, i32 0} ; [ DW_TAG_auto_variable ]
!10 = metadata !{i32 589835, metadata !1, i32 1, i32 16, metadata !2, i32 0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 2, i32 6, metadata !10, null}
!12 = metadata !{i32 2, i32 11, metadata !10, null}
!13 = metadata !{i32 3, i32 2, metadata !10, null}
!14 = metadata !{i32 4, i32 2, metadata !10, null}
!15 = metadata !{i32 5, i32 2, metadata !10, null}
!16 = metadata !{i32 8, i32 2, metadata !17, null}
!17 = metadata !{i32 589835, metadata !6, i32 7, i32 12, metadata !2, i32 1} ; [ DW_TAG_lexical_block ]
