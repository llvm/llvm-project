; RUN: llc < %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin3.0.0-iphoneos"

@length = common global i32 0, align 4            ; <i32*> [#uses=1]

define void @x0(i8* nocapture %buf, i32 %nbytes) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i8* %buf, i64 0, metadata !0, metadata !{!"0x102"}), !dbg !15
  tail call void @llvm.dbg.value(metadata i32 %nbytes, i64 0, metadata !8, metadata !{!"0x102"}), !dbg !16
  %tmp = load i32* @length, !dbg !17              ; <i32> [#uses=3]
  %cmp = icmp eq i32 %tmp, -1, !dbg !17           ; <i1> [#uses=1]
  %cmp.not = xor i1 %cmp, true                    ; <i1> [#uses=1]
  %cmp3 = icmp ult i32 %tmp, %nbytes, !dbg !17    ; <i1> [#uses=1]
  %or.cond = and i1 %cmp.not, %cmp3               ; <i1> [#uses=1]
  tail call void @llvm.dbg.value(metadata i32 %tmp, i64 0, metadata !8, metadata !{!"0x102"}), !dbg !17
  %nbytes.addr.0 = select i1 %or.cond, i32 %tmp, i32 %nbytes ; <i32> [#uses=1]
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !10, metadata !{!"0x102"}), !dbg !19
  br label %while.cond, !dbg !20

while.cond:                                       ; preds = %while.body, %entry
  %0 = phi i32 [ 0, %entry ], [ %inc, %while.body ] ; <i32> [#uses=3]
  %buf.addr.0 = getelementptr i8* %buf, i32 %0    ; <i8*> [#uses=1]
  %cmp7 = icmp ult i32 %0, %nbytes.addr.0, !dbg !20 ; <i1> [#uses=1]
  br i1 %cmp7, label %land.rhs, label %while.end, !dbg !20

land.rhs:                                         ; preds = %while.cond
  %call = tail call i32 @x1() nounwind optsize, !dbg !20 ; <i32> [#uses=2]
  %cmp9 = icmp eq i32 %call, -1, !dbg !20         ; <i1> [#uses=1]
  br i1 %cmp9, label %while.end, label %while.body, !dbg !20

while.body:                                       ; preds = %land.rhs
  %conv = trunc i32 %call to i8, !dbg !21         ; <i8> [#uses=1]
  store i8 %conv, i8* %buf.addr.0, !dbg !21
  %inc = add i32 %0, 1, !dbg !23                  ; <i32> [#uses=1]
  br label %while.cond, !dbg !24

while.end:                                        ; preds = %land.rhs, %while.cond
  ret void, !dbg !25
}

declare i32 @x1() optsize

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.lv.fn = !{!0, !8, !10, !12}
!llvm.dbg.gv = !{!14}

!0 = !{!"0x101\00buf\004\000", !1, !2, !6} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00x0\00x0\00x0\005\000\001\000\006\000\000\000", !26, null, !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !26} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\0012\00clang 2.0\001\00\00\00\00", !26, null, null, null, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !26, !2, null, !5, null} ; [ DW_TAG_subroutine_type ]
!5 = !{null}
!6 = !{!"0xf\00\000\0032\0032\000\000", !26, !2, !7} ; [ DW_TAG_pointer_type ]
!7 = !{!"0x24\00unsigned char\000\008\008\000\000\008", !26, !2} ; [ DW_TAG_base_type ]
!8 = !{!"0x101\00nbytes\004\000", !1, !2, !9} ; [ DW_TAG_arg_variable ]
!9 = !{!"0x24\00unsigned long\000\0032\0032\000\000\007", !26, !2} ; [ DW_TAG_base_type ]
!10 = !{!"0x100\00nread\006\000", !11, !2, !9} ; [ DW_TAG_auto_variable ]
!11 = !{!"0xb\005\001\000", !26, !1} ; [ DW_TAG_lexical_block ]
!12 = !{!"0x100\00c\007\000", !11, !2, !13} ; [ DW_TAG_auto_variable ]
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", !26, !2} ; [ DW_TAG_base_type ]
!14 = !{!"0x34\00length\00length\00length\001\000\001", !2, !2, !13, i32* @length} ; [ DW_TAG_variable ]
!15 = !MDLocation(line: 4, column: 24, scope: !1)
!16 = !MDLocation(line: 4, column: 43, scope: !1)
!17 = !MDLocation(line: 9, column: 2, scope: !11)
!18 = !{i32 0}
!19 = !MDLocation(line: 10, column: 2, scope: !11)
!20 = !MDLocation(line: 11, column: 2, scope: !11)
!21 = !MDLocation(line: 12, column: 3, scope: !22)
!22 = !{!"0xb\0011\0045\000", !26, !11} ; [ DW_TAG_lexical_block ]
!23 = !MDLocation(line: 13, column: 3, scope: !22)
!24 = !MDLocation(line: 14, column: 2, scope: !22)
!25 = !MDLocation(line: 15, column: 1, scope: !11)
!26 = !{!"t.c", !"/private/tmp"}
