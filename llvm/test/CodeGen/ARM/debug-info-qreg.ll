; RUN: llc < %s - | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

;CHECK: sub-register DW_OP_regx
;CHECK-NEXT: ascii
;CHECK-NEXT: DW_OP_piece
;CHECK-NEXT: byte   8
;CHECK-NEXT: sub-register DW_OP_regx
;CHECK-NEXT: ascii
;CHECK-NEXT: DW_OP_piece
;CHECK-NEXT: byte   8

@.str = external constant [13 x i8]

declare <4 x float> @test0001(float) nounwind readnone ssp

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind ssp {
entry:
  br label %for.body9

for.body9:                                        ; preds = %for.body9, %entry
  %add19 = fadd <4 x float> undef, <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, !dbg !39
  br i1 undef, label %for.end54, label %for.body9, !dbg !44

for.end54:                                        ; preds = %for.body9
  tail call void @llvm.dbg.value(metadata <4 x float> %add19, i64 0, metadata !27, metadata !{!"0x102"}), !dbg !39
  %tmp115 = extractelement <4 x float> %add19, i32 1
  %conv6.i75 = fpext float %tmp115 to double, !dbg !45
  %call.i82 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0), double undef, double %conv6.i75, double undef, double undef) nounwind, !dbg !45
  ret i32 0, !dbg !49
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!56}

!0 = !{!"0x2e\00test0001\00test0001\00\003\000\001\000\006\00256\001\003", !54, !1, !3, null, <4 x float> (float)* @test0001, null, null, !51} ; [ DW_TAG_subprogram ] [line 3] [def] [test0001]
!1 = !{!"0x29", !54} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 3.0 (trunk 129915)\001\00\000\00\001", !54, !17, !17, !50, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !54, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x16\00v4f32\0014\000\000\000\000", !54, !2, !6} ; [ DW_TAG_typedef ]
!6 = !{!"0x1\00\000\00128\00128\000\000", !2, null, !7, !8, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 128, offset 0] [from float]
!7 = !{!"0x24\00float\000\0032\0032\000\000\004", null, !2} ; [ DW_TAG_base_type ]
!8 = !{!9}
!9 = !{!"0x21\000\004"}         ; [ DW_TAG_subrange_type ]
!10 = !{!"0x2e\00main\00main\00\0059\000\001\000\006\00256\001\0059", !54, !1, !11, null, i32 (i32, i8**)* @main, null, null, !52} ; [ DW_TAG_subprogram ] [line 59] [def] [main]
!11 = !{!"0x15\00\000\000\000\000\000\000", !54, !1, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!13}
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!14 = !{!"0x2e\00printFV\00printFV\00\0041\001\001\000\006\00256\001\0041", !55, !15, !16, null, null, null, null, !53} ; [ DW_TAG_subprogram ] [line 41] [local] [def] [printFV]
!15 = !{!"0x29", !55} ; [ DW_TAG_file_type ]
!16 = !{!"0x15\00\000\000\000\000\000\000", !55, !15, null, !17, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = !{null}
!18 = !{!"0x101\00a\0016777219\000", !0, !1, !7} ; [ DW_TAG_arg_variable ]
!19 = !{!"0x101\00argc\0016777275\000", !10, !1, !13} ; [ DW_TAG_arg_variable ]
!20 = !{!"0x101\00argv\0033554491\000", !10, !1, !21} ; [ DW_TAG_arg_variable ]
!21 = !{!"0xf\00\000\0032\0032\000\000", null, !2, !22} ; [ DW_TAG_pointer_type ]
!22 = !{!"0xf\00\000\0032\0032\000\000", null, !2, !23} ; [ DW_TAG_pointer_type ]
!23 = !{!"0x24\00char\000\008\008\000\000\006", null, !2} ; [ DW_TAG_base_type ]
!24 = !{!"0x100\00i\0060\000", !25, !1, !13} ; [ DW_TAG_auto_variable ]
!25 = !{!"0xb\0059\0033\0014", !54, !10} ; [ DW_TAG_lexical_block ]
!26 = !{!"0x100\00j\0060\000", !25, !1, !13} ; [ DW_TAG_auto_variable ]
!27 = !{!"0x100\00x\0061\000", !25, !1, !5} ; [ DW_TAG_auto_variable ]
!28 = !{!"0x100\00y\0062\000", !25, !1, !5} ; [ DW_TAG_auto_variable ]
!29 = !{!"0x100\00z\0063\000", !25, !1, !5} ; [ DW_TAG_auto_variable ]
!30 = !{!"0x101\00F\0016777257\000", !14, !15, !31} ; [ DW_TAG_arg_variable ]
!31 = !{!"0xf\00\000\0032\0032\000\000", null, !2, !32} ; [ DW_TAG_pointer_type ]
!32 = !{!"0x16\00FV\0025\000\000\000\000", !55, !2, !33} ; [ DW_TAG_typedef ]
!33 = !{!"0x17\00\0022\00128\00128\000\000\000", !55, !2, i32 0, !34, null} ; [ DW_TAG_union_type ]
!34 = !{!35, !37}
!35 = !{!"0xd\00V\0023\00128\00128\000\000", !55, !15, !36} ; [ DW_TAG_member ]
!36 = !{!"0x16\00v4sf\003\000\000\000\000", !55, !2, !6} ; [ DW_TAG_typedef ]
!37 = !{!"0xd\00A\0024\00128\0032\000\000", !55, !15, !38} ; [ DW_TAG_member ]
!38 = !{!"0x1\00\000\00128\0032\000\000", null, !2, !7, !8, i32 0, i32 0} ; [ DW_TAG_array_type ]
!39 = !MDLocation(line: 79, column: 7, scope: !40)
!40 = !{!"0xb\0075\0035\0018", !54, !41} ; [ DW_TAG_lexical_block ]
!41 = !{!"0xb\0075\005\0017", !54, !42} ; [ DW_TAG_lexical_block ]
!42 = !{!"0xb\0071\0032\0016", !54, !43} ; [ DW_TAG_lexical_block ]
!43 = !{!"0xb\0071\003\0015", !54, !25} ; [ DW_TAG_lexical_block ]
!44 = !MDLocation(line: 75, column: 5, scope: !42)
!45 = !MDLocation(line: 42, column: 2, scope: !46, inlinedAt: !48)
!46 = !{!"0xb\0042\002\0020", !55, !47} ; [ DW_TAG_lexical_block ]
!47 = !{!"0xb\0041\0028\0019", !55, !14} ; [ DW_TAG_lexical_block ]
!48 = !MDLocation(line: 95, column: 3, scope: !25)
!49 = !MDLocation(line: 99, column: 3, scope: !25)
!50 = !{!0, !10, !14}
!51 = !{!18}
!52 = !{!19, !20, !24, !26, !27, !28, !29}
!53 = !{!30}
!54 = !{!"build2.c", !"/private/tmp"}
!55 = !{!"/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/UnitTests/Vector/helpers.h", !"/private/tmp"}
!56 = !{i32 1, !"Debug Info Version", i32 2}
