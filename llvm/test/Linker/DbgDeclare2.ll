; This file is used by 2011-08-04-DebugLoc.ll, so it doesn't actually do anything itself
;
; RUN: true

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

define void @test(i32 %argc, i8** %argv) uwtable ssp {
entry:
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %i = alloca i32, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !14), !dbg !15
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !16), !dbg !15
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !17), !dbg !20
  store i32 0, i32* %i, align 4, !dbg !20
  br label %for.cond, !dbg !20

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32* %i, align 4, !dbg !20
  %1 = load i32* %argc.addr, align 4, !dbg !20
  %cmp = icmp slt i32 %0, %1, !dbg !20
  br i1 %cmp, label %for.body, label %for.end, !dbg !20

for.body:                                         ; preds = %for.cond
  %2 = load i32* %i, align 4, !dbg !21
  %idxprom = sext i32 %2 to i64, !dbg !21
  %3 = load i8*** %argv.addr, align 8, !dbg !21
  %arrayidx = getelementptr inbounds i8** %3, i64 %idxprom, !dbg !21
  %4 = load i8** %arrayidx, align 8, !dbg !21
  %call = call i32 @puts(i8* %4), !dbg !21
  br label %for.inc, !dbg !23

for.inc:                                          ; preds = %for.body
  %5 = load i32* %i, align 4, !dbg !20
  %inc = add nsw i32 %5, 1, !dbg !20
  store i32 %inc, i32* %i, align 4, !dbg !20
  br label %for.cond, !dbg !20

for.end:                                          ; preds = %for.cond
  ret void, !dbg !24
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i32 @puts(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = metadata !{i32 786449, metadata !25, i32 4, metadata !"clang version 3.3 (trunk 173515)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !26, null, metadata !"print_args", metadata !"print_args", metadata !"test", i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32, i8**)* @test, null, null, metadata !1, i32 5} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !26} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9, metadata !10}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 786447, null, null, null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 786447, null, null, null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{i32 786470, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !13} ; [ DW_TAG_const_type ]
!13 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 786689, metadata !5, metadata !"argc", metadata !6, i32 16777220, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 4, i32 0, metadata !5, null}
!16 = metadata !{i32 786689, metadata !5, metadata !"argv", metadata !6, i32 33554436, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 786688, metadata !18, metadata !"i", metadata !6, i32 6, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!18 = metadata !{i32 786443, metadata !26, metadata !19, i32 6, i32 0, i32 1} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 786443, metadata !26, metadata !5, i32 5, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!20 = metadata !{i32 6, i32 0, metadata !18, null}
!21 = metadata !{i32 8, i32 0, metadata !22, null}
!22 = metadata !{i32 786443, metadata !26, metadata !18, i32 7, i32 0, i32 2} ; [ DW_TAG_lexical_block ]
!23 = metadata !{i32 9, i32 0, metadata !22, null}
!24 = metadata !{i32 10, i32 0, metadata !19, null}
!25 = metadata !{metadata !"main.cpp", metadata !"/private/tmp"}
!26 = metadata !{metadata !"test.cpp", metadata !"/private/tmp"}
!27 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
