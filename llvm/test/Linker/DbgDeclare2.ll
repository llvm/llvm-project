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
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !14, metadata !{!"0x102"}), !dbg !15
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !16, metadata !{!"0x102"}), !dbg !15
  call void @llvm.dbg.declare(metadata i32* %i, metadata !17, metadata !{!"0x102"}), !dbg !20
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

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @puts(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = !{!"0x11\004\00clang version 3.3 (trunk 173515)\001\00\000\00\000", !25, !2, !2, !3, !2, null} ; [ DW_TAG_compile_unit ]
!1 = !{!2}
!2 = !{i32 0}
!3 = !{!5}
!5 = !{!"0x2e\00print_args\00print_args\00test\004\000\001\000\006\00256\000\005", !26, null, !7, null, void (i32, i8**)* @test, null, null, !1} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !26} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null, !9, !10}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ]
!11 = !{!"0xf\00\000\0064\0064\000\000", null, null, !12} ; [ DW_TAG_pointer_type ]
!12 = !{!"0x26\00\000\000\000\000\000", null, null, !13} ; [ DW_TAG_const_type ]
!13 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!14 = !{!"0x101\00argc\0016777220\000", !5, !6, !9} ; [ DW_TAG_arg_variable ]
!15 = !MDLocation(line: 4, scope: !5)
!16 = !{!"0x101\00argv\0033554436\000", !5, !6, !10} ; [ DW_TAG_arg_variable ]
!17 = !{!"0x100\00i\006\000", !18, !6, !9} ; [ DW_TAG_auto_variable ]
!18 = !{!"0xb\006\000\001", !26, !19} ; [ DW_TAG_lexical_block ]
!19 = !{!"0xb\005\000\000", !26, !5} ; [ DW_TAG_lexical_block ]
!20 = !MDLocation(line: 6, scope: !18)
!21 = !MDLocation(line: 8, scope: !22)
!22 = !{!"0xb\007\000\002", !26, !18} ; [ DW_TAG_lexical_block ]
!23 = !MDLocation(line: 9, scope: !22)
!24 = !MDLocation(line: 10, scope: !19)
!25 = !{!"main.cpp", !"/private/tmp"}
!26 = !{!"test.cpp", !"/private/tmp"}
!27 = !{i32 1, !"Debug Info Version", i32 2}
