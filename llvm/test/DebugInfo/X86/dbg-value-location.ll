; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; RUN: llc -filetype=obj %s -regalloc=basic -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"
; Test that the type for the formal parameter "var" makes it into the debug info.
; rdar://8950491

;CHECK: DW_TAG_formal_parameter
;CHECK-NEXT: DW_AT_location
;CHECK-NEXT: DW_AT_name {{.*}} "var"
;CHECK-NEXT: DW_AT_decl_file
;CHECK-NEXT: DW_AT_decl_line
;CHECK-NEXT: DW_AT_type

@dfm = external global i32, align 4

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @foo(i32 %dev, i64 %cmd, i8* %data, i32 %data2) nounwind optsize ssp {
entry:
  call void @llvm.dbg.value(metadata i32 %dev, i64 0, metadata !12, metadata !{!"0x102"}), !dbg !13
  %tmp.i = load i32* @dfm, align 4, !dbg !14
  %cmp.i = icmp eq i32 %tmp.i, 0, !dbg !14
  br i1 %cmp.i, label %if.else, label %if.end.i, !dbg !14

if.end.i:                                         ; preds = %entry
  switch i64 %cmd, label %if.then [
    i64 2147772420, label %bb.i
    i64 536897538, label %bb116.i
  ], !dbg !22

bb.i:                                             ; preds = %if.end.i
  unreachable

bb116.i:                                          ; preds = %if.end.i
  unreachable

if.then:                                          ; preds = %if.end.i
  ret i32 undef, !dbg !23

if.else:                                          ; preds = %entry
  ret i32 0
}

declare hidden fastcc i32 @bar(i32, i32* nocapture) nounwind optsize ssp
declare hidden fastcc i32 @bar2(i32) nounwind optsize ssp
declare hidden fastcc i32 @bar3(i32) nounwind optsize ssp
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!29}

!0 = !{!"0x2e\00foo\00foo\00\0019510\000\001\000\006\00256\001\0019510", !26, !1, !3, null, i32 (i32, i64, i8*, i32)* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 19510] [def] [foo]
!1 = !{!"0x29", !26} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 124753)\001\00\000\00\000", !27, !28, !28, !24, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !26, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00bar3\00bar3\00\0014827\001\001\000\006\00256\001\000", !26, !1, !3, null, i32 (i32)* @bar3, null, null, null} ; [ DW_TAG_subprogram ] [line 14827] [local] [def] [scope 0] [bar3]
!7 = !{!"0x2e\00bar2\00bar2\00\0015397\001\001\000\006\00256\001\000", !26, !1, !3, null, i32 (i32)* @bar2, null, null, null} ; [ DW_TAG_subprogram ] [line 15397] [local] [def] [scope 0] [bar2]
!8 = !{!"0x2e\00bar\00bar\00\0012382\001\001\000\006\00256\001\000", !26, !1, !9, null, i32 (i32, i32*)* @bar, null, null, null} ; [ DW_TAG_subprogram ] [line 12382] [local] [def] [scope 0] [bar]
!9 = !{!"0x15\00\000\000\000\000\000\000", !26, !1, null, !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = !{!11}
!11 = !{!"0x24\00unsigned char\000\008\008\000\000\008", null, !2} ; [ DW_TAG_base_type ]
!12 = !{!"0x101\00var\0019509\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!13 = !MDLocation(line: 19509, column: 20, scope: !0)
!14 = !MDLocation(line: 18091, column: 2, scope: !15, inlinedAt: !17)
!15 = !{!"0xb\0018086\001\00748", !26, !16} ; [ DW_TAG_lexical_block ]
!16 = !{!"0x2e\00foo_bar\00foo_bar\00\0018086\001\001\000\006\00256\001\000", !26, !1, !3, null, null, null, null, null} ; [ DW_TAG_subprogram ] [line 18086] [local] [def] [scope 0] [foo_bar]
!17 = !MDLocation(line: 19514, column: 2, scope: !18)
!18 = !{!"0xb\0019510\001\0099", !26, !0} ; [ DW_TAG_lexical_block ]
!22 = !MDLocation(line: 18094, column: 2, scope: !15, inlinedAt: !17)
!23 = !MDLocation(line: 19524, column: 1, scope: !18)
!24 = !{!0, !6, !7, !8, !16}
!25 = !{!"0x29", !27} ; [ DW_TAG_file_type ]
!26 = !{!"/tmp/f.c", !"/tmp"}
!27 = !{!"f.i", !"/tmp"}
!28 = !{i32 0}
!29 = !{i32 1, !"Debug Info Version", i32 2}
