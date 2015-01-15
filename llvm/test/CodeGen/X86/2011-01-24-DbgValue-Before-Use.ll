; RUN: llc < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; RUN: llc < %s -filetype=obj -regalloc=basic | llvm-dwarfdump -debug-dump=info -  | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; Check debug info for variable z_s
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_variable
; CHECK: DW_TAG_variable
; CHECK-NEXT:   DW_AT_location
; CHECK-NEXT:   DW_AT_name {{.*}} "z_s"
; CHECK-NEXT:   DW_AT_decl_file
; CHECK-NEXT:   DW_AT_decl_line
; CHECK-NEXT:   DW_AT_type{{.*}}{[[TYPE:.*]]}
; CHECK: [[TYPE]]:
; CHECK-NEXT: DW_AT_name {{.*}} "int"


@.str1 = private unnamed_addr constant [14 x i8] c"m=%u, z_s=%d\0A\00"
@str = internal constant [21 x i8] c"Failing test vector:\00"

define i64 @gcd(i64 %a, i64 %b) nounwind readnone optsize noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata i64 %a, i64 0, metadata !10, metadata !{!"0x102"}), !dbg !18
  tail call void @llvm.dbg.value(metadata i64 %b, i64 0, metadata !11, metadata !{!"0x102"}), !dbg !19
  br label %while.body, !dbg !20

while.body:                                       ; preds = %while.body, %entry
  %b.addr.0 = phi i64 [ %b, %entry ], [ %rem, %while.body ]
  %a.addr.0 = phi i64 [ %a, %entry ], [ %b.addr.0, %while.body ]
  %rem = srem i64 %a.addr.0, %b.addr.0, !dbg !21
  %cmp = icmp eq i64 %rem, 0, !dbg !23
  br i1 %cmp, label %if.then, label %while.body, !dbg !23

if.then:                                          ; preds = %while.body
  tail call void @llvm.dbg.value(metadata i64 %rem, i64 0, metadata !12, metadata !{!"0x102"}), !dbg !21
  ret i64 %b.addr.0, !dbg !23
}

define i32 @main() nounwind optsize ssp {
entry:
  %call = tail call i32 @rand() nounwind optsize, !dbg !24
  tail call void @llvm.dbg.value(metadata i32 %call, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !24
  %cmp = icmp ugt i32 %call, 21, !dbg !25
  br i1 %cmp, label %cond.true, label %cond.end, !dbg !25

cond.true:                                        ; preds = %entry
  %call1 = tail call i32 @rand() nounwind optsize, !dbg !25
  br label %cond.end, !dbg !25

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %call1, %cond.true ], [ %call, %entry ], !dbg !25
  tail call void @llvm.dbg.value(metadata i32 %cond, i64 0, metadata !17, metadata !{!"0x102"}), !dbg !25
  %conv = sext i32 %cond to i64, !dbg !26
  %conv5 = zext i32 %call to i64, !dbg !26
  %call6 = tail call i64 @gcd(i64 %conv, i64 %conv5) optsize, !dbg !26
  %cmp7 = icmp eq i64 %call6, 0
  br i1 %cmp7, label %return, label %if.then, !dbg !26

if.then:                                          ; preds = %cond.end
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([21 x i8]* @str, i64 0, i64 0))
  %call12 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([14 x i8]* @.str1, i64 0, i64 0), i32 %call, i32 %cond) nounwind optsize, !dbg !26
  ret i32 1, !dbg !27

return:                                           ; preds = %cond.end
  ret i32 0, !dbg !27
}

declare i32 @rand() optsize

declare i32 @printf(i8* nocapture, ...) nounwind optsize

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

declare i32 @puts(i8* nocapture) nounwind

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!33}

!0 = !{!"0x2e\00gcd\00gcd\00\005\000\001\000\006\00256\001\000", !31, !1, !3, null, i64 (i64, i64)* @gcd, null, null, !29} ; [ DW_TAG_subprogram ] [line 5] [def] [scope 0] [gcd]
!1 = !{!"0x29", !31} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 124117)\001\00\000\00\001", !31, !32, !32, !28, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !31, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00long int\000\0064\0064\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00main\00main\00\0025\000\001\000\006\000\001\000", !31, !1, !7, null, i32 ()* @main, null, null, !30} ; [ DW_TAG_subprogram ] [line 25] [def] [scope 0] [main]
!7 = !{!"0x15\00\000\000\000\000\000\000", !31, !1, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!10 = !{!"0x101\00a\005\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!11 = !{!"0x101\00b\005\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!12 = !{!"0x100\00c\006\000", !13, !1, !5} ; [ DW_TAG_auto_variable ]
!13 = !{!"0xb\005\0052\000", !31, !0} ; [ DW_TAG_lexical_block ]
!14 = !{!"0x100\00m\0026\000", !15, !1, !16} ; [ DW_TAG_auto_variable ]
!15 = !{!"0xb\0025\0012\002", !31, !6} ; [ DW_TAG_lexical_block ]
!16 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, !2} ; [ DW_TAG_base_type ]
!17 = !{!"0x100\00z_s\0027\000", !15, !1, !9} ; [ DW_TAG_auto_variable ]
!18 = !MDLocation(line: 5, column: 41, scope: !0)
!19 = !MDLocation(line: 5, column: 49, scope: !0)
!20 = !MDLocation(line: 7, column: 5, scope: !13)
!21 = !MDLocation(line: 8, column: 9, scope: !22)
!22 = !{!"0xb\007\0014\001", !31, !13} ; [ DW_TAG_lexical_block ]
!23 = !MDLocation(line: 9, column: 9, scope: !22)
!24 = !MDLocation(line: 26, column: 38, scope: !15)
!25 = !MDLocation(line: 27, column: 38, scope: !15)
!26 = !MDLocation(line: 28, column: 9, scope: !15)
!27 = !MDLocation(line: 30, column: 1, scope: !15)
!28 = !{!0, !6}
!29 = !{!10, !11, !12}
!30 = !{!14, !17}
!31 = !{!"rem_small.c", !"/private/tmp"}
!32 = !{i32 0}
!33 = !{i32 1, !"Debug Info Version", i32 2}
