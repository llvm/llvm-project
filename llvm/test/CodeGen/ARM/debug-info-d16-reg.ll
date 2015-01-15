; RUN: llc < %s | FileCheck %s
; Radar 9309221
; Test dwarf reg no for d16
;CHECK: DW_OP_regx
;CHECK-NEXT: 272

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

@.str = private unnamed_addr constant [11 x i8] c"%p %lf %c\0A\00", align 4
@.str1 = private unnamed_addr constant [6 x i8] c"point\00", align 4

define i32 @inlineprinter(i8* %ptr, double %val, i8 zeroext %c) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i8* %ptr, i64 0, metadata !19, metadata !{!"0x102"}), !dbg !26
  tail call void @llvm.dbg.value(metadata double %val, i64 0, metadata !20, metadata !{!"0x102"}), !dbg !26
  tail call void @llvm.dbg.value(metadata i8 %c, i64 0, metadata !21, metadata !{!"0x102"}), !dbg !26
  %0 = zext i8 %c to i32, !dbg !27
  %1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %val, i32 %0) nounwind, !dbg !27
  ret i32 0, !dbg !29
}

define i32 @printer(i8* %ptr, double %val, i8 zeroext %c) nounwind optsize noinline {
entry:
  tail call void @llvm.dbg.value(metadata i8* %ptr, i64 0, metadata !16, metadata !{!"0x102"}), !dbg !30
  tail call void @llvm.dbg.value(metadata double %val, i64 0, metadata !17, metadata !{!"0x102"}), !dbg !30
  tail call void @llvm.dbg.value(metadata i8 %c, i64 0, metadata !18, metadata !{!"0x102"}), !dbg !30
  %0 = zext i8 %c to i32, !dbg !31
  %1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %val, i32 %0) nounwind, !dbg !31
  ret i32 0, !dbg !33
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !22, metadata !{!"0x102"}), !dbg !34
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !23, metadata !{!"0x102"}), !dbg !34
  %0 = sitofp i32 %argc to double, !dbg !35
  %1 = fadd double %0, 5.555552e+05, !dbg !35
  tail call void @llvm.dbg.value(metadata double %1, i64 0, metadata !24, metadata !{!"0x102"}), !dbg !35
  %2 = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @.str1, i32 0, i32 0)) nounwind, !dbg !36
  %3 = getelementptr inbounds i8* bitcast (i32 (i32, i8**)* @main to i8*), i32 %argc, !dbg !37
  %4 = trunc i32 %argc to i8, !dbg !37
  %5 = add i8 %4, 97, !dbg !37
  tail call void @llvm.dbg.value(metadata i8* %3, i64 0, metadata !19, metadata !{!"0x102"}) nounwind, !dbg !38
  tail call void @llvm.dbg.value(metadata double %1, i64 0, metadata !20, metadata !{!"0x102"}) nounwind, !dbg !38
  tail call void @llvm.dbg.value(metadata i8 %5, i64 0, metadata !21, metadata !{!"0x102"}) nounwind, !dbg !38
  %6 = zext i8 %5 to i32, !dbg !39
  %7 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %3, double %1, i32 %6) nounwind, !dbg !39
  %8 = tail call i32 @printer(i8* %3, double %1, i8 zeroext %5) nounwind, !dbg !40
  ret i32 0, !dbg !41
}

declare i32 @puts(i8* nocapture) nounwind

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!48}

!0 = !{!"0x2e\00printer\00printer\00printer\0012\000\001\000\006\00256\001\0012", !46, !1, !3, null, i32 (i8*, double, i8)* @printer, null, null, !43} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !46} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\001\00(LLVM build 00)\001\00\000\00\001", !46, !47, !47, !42, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !46, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5, !6, !7, !8}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", !46, !1} ; [ DW_TAG_base_type ]
!6 = !{!"0xf\00\000\0032\0032\000\000", !46, !1, null} ; [ DW_TAG_pointer_type ]
!7 = !{!"0x24\00double\000\0064\0032\000\000\004", !46, !1} ; [ DW_TAG_base_type ]
!8 = !{!"0x24\00unsigned char\000\008\008\000\000\008", !46, !1} ; [ DW_TAG_base_type ]
!9 = !{!"0x2e\00inlineprinter\00inlineprinter\00inlineprinter\005\000\001\000\006\00256\001\005", !46, !1, !3, null, i32 (i8*, double, i8)* @inlineprinter, null, null, !44} ; [ DW_TAG_subprogram ]
!10 = !{!"0x2e\00main\00main\00main\0018\000\001\000\006\00256\001\0018", !46, !1, !11, null, i32 (i32, i8**)* @main, null, null, !45} ; [ DW_TAG_subprogram ]
!11 = !{!"0x15\00\000\000\000\000\000\000", !46, !1, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!5, !5, !13}
!13 = !{!"0xf\00\000\0032\0032\000\000", !46, !1, !14} ; [ DW_TAG_pointer_type ]
!14 = !{!"0xf\00\000\0032\0032\000\000", !46, !1, !15} ; [ DW_TAG_pointer_type ]
!15 = !{!"0x24\00char\000\008\008\000\000\006", !46, !1} ; [ DW_TAG_base_type ]
!16 = !{!"0x101\00ptr\0011\000", !0, !1, !6} ; [ DW_TAG_arg_variable ]
!17 = !{!"0x101\00val\0011\000", !0, !1, !7} ; [ DW_TAG_arg_variable ]
!18 = !{!"0x101\00c\0011\000", !0, !1, !8} ; [ DW_TAG_arg_variable ]
!19 = !{!"0x101\00ptr\004\000", !9, !1, !6} ; [ DW_TAG_arg_variable ]
!20 = !{!"0x101\00val\004\000", !9, !1, !7} ; [ DW_TAG_arg_variable ]
!21 = !{!"0x101\00c\004\000", !9, !1, !8} ; [ DW_TAG_arg_variable ]
!22 = !{!"0x101\00argc\0017\000", !10, !1, !5} ; [ DW_TAG_arg_variable ]
!23 = !{!"0x101\00argv\0017\000", !10, !1, !13} ; [ DW_TAG_arg_variable ]
!24 = !{!"0x100\00dval\0019\000", !25, !1, !7} ; [ DW_TAG_auto_variable ]
!25 = !{!"0xb\0018\000\002", !46, !10} ; [ DW_TAG_lexical_block ]
!26 = !MDLocation(line: 4, scope: !9)
!27 = !MDLocation(line: 6, scope: !28)
!28 = !{!"0xb\005\000\001", !46, !9} ; [ DW_TAG_lexical_block ]
!29 = !MDLocation(line: 7, scope: !28)
!30 = !MDLocation(line: 11, scope: !0)
!31 = !MDLocation(line: 13, scope: !32)
!32 = !{!"0xb\0012\000\000", !46, !0} ; [ DW_TAG_lexical_block ]
!33 = !MDLocation(line: 14, scope: !32)
!34 = !MDLocation(line: 17, scope: !10)
!35 = !MDLocation(line: 19, scope: !25)
!36 = !MDLocation(line: 20, scope: !25)
!37 = !MDLocation(line: 21, scope: !25)
!38 = !MDLocation(line: 4, scope: !9, inlinedAt: !37)
!39 = !MDLocation(line: 6, scope: !28, inlinedAt: !37)
!40 = !MDLocation(line: 22, scope: !25)
!41 = !MDLocation(line: 23, scope: !25)
!42 = !{!0, !9, !10}
!43 = !{!16, !17, !18}
!44 = !{!19, !20, !21}
!45 = !{!22, !23, !24}
!46 = !{!"a.c", !"/tmp/"}
!47 = !{i32 0}
!48 = !{i32 1, !"Debug Info Version", i32 2}
