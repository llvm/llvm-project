; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

; Check that the line table starts at 7, not 4, but that the first
; statement isn't until line 8.

; CHECK-NOT: 0x0000000000000000      7      0      1   0  is_stmt
; CHECK: 0x0000000000000000      7      0      1   0
; CHECK: 0x0000000000000004      8     18      1   0  is_stmt prologue_end

define i32 @callee(i32 %x) nounwind uwtable ssp {
entry:
  %x.addr = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !12), !dbg !13
  call void @llvm.dbg.declare(metadata !{i32* %y}, metadata !14), !dbg !16
  %0 = load i32* %x.addr, align 4, !dbg !17
  %1 = load i32* %x.addr, align 4, !dbg !17
  %mul = mul nsw i32 %0, %1, !dbg !17
  store i32 %mul, i32* %y, align 4, !dbg !17
  %2 = load i32* %y, align 4, !dbg !18
  %sub = sub nsw i32 %2, 2, !dbg !18
  ret i32 %sub, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 12, metadata !"ending-run.c", metadata !"/Users/echristo/tmp", metadata !"clang version 3.1 (trunk 153921) (llvm/trunk 153916)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786478, i32 0, metadata !6, metadata !"callee", metadata !"callee", metadata !"", metadata !6, i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 (i32)* @callee, null, null, metadata !10, i32 7} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !"ending-run.c", metadata !"/Users/echristo/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!12 = metadata !{i32 786689, metadata !5, metadata !"x", metadata !6, i32 16777221, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!13 = metadata !{i32 5, i32 5, metadata !5, null}
!14 = metadata !{i32 786688, metadata !15, metadata !"y", metadata !6, i32 8, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!15 = metadata !{i32 786443, metadata !5, i32 7, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 8, i32 9, metadata !15, null}
!17 = metadata !{i32 8, i32 18, metadata !15, null}
!18 = metadata !{i32 9, i32 5, metadata !15, null}
