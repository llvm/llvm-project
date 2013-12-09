; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s -check-prefix=DW-CHECK

; DW-CHECK: DW_AT_name [DW_FORM_strp]  ( .debug_str[0x00000067] = "vla")
; FIXME: The location here needs to be fixed, but llvm-dwarfdump doesn't handle
; DW_AT_location lists yet.
; DW-CHECK: DW_AT_location [DW_FORM_sec_offset]                      (0x00000000)

; Unfortunately llvm-dwarfdump can't unparse a list of DW_AT_locations
; right now, so we check the asm output:
; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o - -filetype=asm | FileCheck %s -check-prefix=ASM-CHECK
; vla should have a register-indirect address at one point.
; ASM-CHECK: DEBUG_VALUE: vla <- RCX
; ASM-CHECK: DW_OP_breg2

define void @testVLAwithSize(i32 %s) nounwind uwtable ssp {
entry:
  %s.addr = alloca i32, align 4
  %saved_stack = alloca i8*
  %i = alloca i32, align 4
  store i32 %s, i32* %s.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %s.addr}, metadata !10), !dbg !11
  %0 = load i32* %s.addr, align 4, !dbg !12
  %1 = zext i32 %0 to i64, !dbg !12
  %2 = call i8* @llvm.stacksave(), !dbg !12
  store i8* %2, i8** %saved_stack, !dbg !12
  %vla = alloca i32, i64 %1, align 16, !dbg !12
  call void @llvm.dbg.declare(metadata !{i32* %vla}, metadata !14), !dbg !18
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !19), !dbg !20
  store i32 0, i32* %i, align 4, !dbg !21
  br label %for.cond, !dbg !21

for.cond:                                         ; preds = %for.inc, %entry
  %3 = load i32* %i, align 4, !dbg !21
  %4 = load i32* %s.addr, align 4, !dbg !21
  %cmp = icmp slt i32 %3, %4, !dbg !21
  br i1 %cmp, label %for.body, label %for.end, !dbg !21

for.body:                                         ; preds = %for.cond
  %5 = load i32* %i, align 4, !dbg !23
  %6 = load i32* %i, align 4, !dbg !23
  %mul = mul nsw i32 %5, %6, !dbg !23
  %7 = load i32* %i, align 4, !dbg !23
  %idxprom = sext i32 %7 to i64, !dbg !23
  %arrayidx = getelementptr inbounds i32* %vla, i64 %idxprom, !dbg !23
  store i32 %mul, i32* %arrayidx, align 4, !dbg !23
  br label %for.inc, !dbg !25

for.inc:                                          ; preds = %for.body
  %8 = load i32* %i, align 4, !dbg !26
  %inc = add nsw i32 %8, 1, !dbg !26
  store i32 %inc, i32* %i, align 4, !dbg !26
  br label %for.cond, !dbg !26

for.end:                                          ; preds = %for.cond
  %9 = load i8** %saved_stack, !dbg !27
  call void @llvm.stackrestore(i8* %9), !dbg !27
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i8* @llvm.stacksave() nounwind

declare void @llvm.stackrestore(i8*) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = metadata !{i32 786449, metadata !28, i32 12, metadata !"clang version 3.2 (trunk 156005) (llvm/trunk 156000)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !28, metadata !6, metadata !"testVLAwithSize", metadata !"testVLAwithSize", metadata !"", i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @testVLAwithSize, null, null, metadata !1, i32 2} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !28} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 786689, metadata !5, metadata !"s", metadata !6, i32 16777217, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 1, i32 26, metadata !5, null}
!12 = metadata !{i32 3, i32 13, metadata !13, null}
!13 = metadata !{i32 786443, metadata !28, metadata !5, i32 2, i32 1, i32 0} ; [ DW_TAG_lexical_block ]
!14 = metadata !{i32 786688, metadata !13, metadata !"vla", metadata !6, i32 3, metadata !15, i32 8192, i32 0, i64 2} ; [ DW_TAG_auto_variable ]
!15 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 0, i64 32, i32 0, i32 0, metadata !9, metadata !16, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 786465, i64 0, i64 -1}        ; [ DW_TAG_subrange_type ]
!18 = metadata !{i32 3, i32 7, metadata !13, null}
!19 = metadata !{i32 786688, metadata !13, metadata !"i", metadata !6, i32 4, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!20 = metadata !{i32 4, i32 7, metadata !13, null}
!21 = metadata !{i32 5, i32 8, metadata !22, null}
!22 = metadata !{i32 786443, metadata !28, metadata !13, i32 5, i32 3, i32 1} ; [ DW_TAG_lexical_block ]
!23 = metadata !{i32 6, i32 5, metadata !24, null}
!24 = metadata !{i32 786443, metadata !28, metadata !22, i32 5, i32 27, i32 2} ; [ DW_TAG_lexical_block ]
!25 = metadata !{i32 7, i32 3, metadata !24, null}
!26 = metadata !{i32 5, i32 22, metadata !22, null}
!27 = metadata !{i32 8, i32 1, metadata !13, null}
!28 = metadata !{metadata !"bar.c", metadata !"/Users/echristo/tmp"}
!29 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
