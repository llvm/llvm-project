; RUN: llc < %s | FileCheck %s
; RUN: llc < %s -regalloc=basic | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"
;Radar 8950491

;CHECK: .long Lset5
;CHECK-NEXT:        ## DW_AT_decl_file
;CHECK-NEXT:        ## DW_AT_decl_line
;CHECK-NEXT:        ## DW_AT_type
;CHECK-NEXT:        ## DW_AT_location

@dfm = external global i32, align 4

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @foo(i32 %dev, i64 %cmd, i8* %data, i32 %data2) nounwind optsize ssp {
entry:
  call void @llvm.dbg.value(metadata !{i32 %dev}, i64 0, metadata !12), !dbg !13
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
declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!29}

!0 = metadata !{i32 786478, metadata !26, metadata !1, metadata !"foo", metadata !"foo", metadata !"", i32 19510, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32, i64, i8*, i32)* @foo, null, null, null, i32 19510} ; [ DW_TAG_subprogram ] [line 19510] [def] [foo]
!1 = metadata !{i32 786473, metadata !26} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !27, i32 12, metadata !"clang version 2.9 (trunk 124753)", i1 true, metadata !"", i32 0, metadata !28, metadata !28, metadata !24, null,  null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !26, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, null, metadata !2, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 786478, metadata !26, metadata !1, metadata !"bar3", metadata !"bar3", metadata !"", i32 14827, metadata !3, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32)* @bar3, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 14827] [local] [def] [scope 0] [bar3]
!7 = metadata !{i32 786478, metadata !26, metadata !1, metadata !"bar2", metadata !"bar2", metadata !"", i32 15397, metadata !3, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32)* @bar2, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 15397] [local] [def] [scope 0] [bar2]
!8 = metadata !{i32 786478, metadata !26, metadata !1, metadata !"bar", metadata !"bar", metadata !"", i32 12382, metadata !9, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32, i32*)* @bar, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 12382] [local] [def] [scope 0] [bar]
!9 = metadata !{i32 786453, metadata !26, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786468, null, metadata !2, metadata !"unsigned char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 8} ; [ DW_TAG_base_type ]
!12 = metadata !{i32 786689, metadata !0, metadata !"var", metadata !1, i32 19509, metadata !5, i32 0, null} ; [ DW_TAG_arg_variable ]
!13 = metadata !{i32 19509, i32 20, metadata !0, null}
!14 = metadata !{i32 18091, i32 2, metadata !15, metadata !17}
!15 = metadata !{i32 786443, metadata !26, metadata !16, i32 18086, i32 1, i32 748} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 786478, metadata !26, metadata !1, metadata !"foo_bar", metadata !"foo_bar", metadata !"", i32 18086, metadata !3, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, null, null, i32 0} ; [ DW_TAG_subprogram ] [line 18086] [local] [def] [scope 0] [foo_bar]
!17 = metadata !{i32 19514, i32 2, metadata !18, null}
!18 = metadata !{i32 786443, metadata !26, metadata !0, i32 19510, i32 1, i32 99} ; [ DW_TAG_lexical_block ]
!22 = metadata !{i32 18094, i32 2, metadata !15, metadata !17}
!23 = metadata !{i32 19524, i32 1, metadata !18, null}
!24 = metadata !{metadata !0, metadata !6, metadata !7, metadata !8}
!25 = metadata !{i32 786473, metadata !27} ; [ DW_TAG_file_type ]
!26 = metadata !{metadata !"/tmp/f.c", metadata !"/tmp"}
!27 = metadata !{metadata !"f.i", metadata !"/tmp"}
!28 = metadata !{i32 0}
!29 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
