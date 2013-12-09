; RUN: llc -mtriple=s390x-linux-gnu -disable-fp-elim < %s | FileCheck %s
;
; This is a regression test making sure the location of variables is correct in
; debugging information, even if they're addressed via the frame pointer.
; A copy of the AArch64 test, commandeered for SystemZ.
;
; First make sure main_arr is where we expect it: %r11 + 164
;
; CHECK: main:
; CHECK: aghi    %r15, -568
; CHECK: la      %r2, 164(%r11)
; CHECK: brasl   %r14, populate_array@PLT
;
; CHECK: .Linfo_string7:
; CHECK-NEXT: main_arr
;
; Now check that the debugging information reflects this:
; CHECK: DW_TAG_variable
; CHECK-NEXT: .long .Linfo_string7
;
; Rather hard-coded, but 145 => DW_OP_fbreg and the .ascii is the sleb128
; encoding of 164:
; CHECK: DW_AT_location
; CHECK-NEXT: .byte 145
; CHECK-NEXT: .ascii "\244\001"
;


@.str = private unnamed_addr constant [13 x i8] c"Total is %d\0A\00", align 2

declare void @populate_array(i32*, i32) nounwind

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i32 @sum_array(i32*, i32) nounwind

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  %main_arr = alloca [100 x i32], align 4
  %val = alloca i32, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata !{[100 x i32]* %main_arr}, metadata !17), !dbg !22
  call void @llvm.dbg.declare(metadata !{i32* %val}, metadata !23), !dbg !24
  %arraydecay = getelementptr inbounds [100 x i32]* %main_arr, i32 0, i32 0, !dbg !25
  call void @populate_array(i32* %arraydecay, i32 100), !dbg !25
  %arraydecay1 = getelementptr inbounds [100 x i32]* %main_arr, i32 0, i32 0, !dbg !26
  %call = call i32 @sum_array(i32* %arraydecay1, i32 100), !dbg !26
  store i32 %call, i32* %val, align 4, !dbg !26
  %0 = load i32* %val, align 4, !dbg !27
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0), i32 %0), !dbg !27
  ret i32 0, !dbg !28
}

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!30}

!0 = metadata !{i32 786449, metadata !29, i32 12, metadata !"clang version 3.2 ", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ] [/home/timnor01/a64-trunk/build/simple.c] [DW_LANG_C99]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !11, metadata !14}
!5 = metadata !{i32 786478, metadata !29, metadata !6, metadata !"populate_array", metadata !"populate_array", metadata !"", i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32*, i32)* @populate_array, null, null, metadata !1, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [populate_array]
!6 = metadata !{i32 786473, metadata !29} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9, metadata !10}
!9 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!10 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = metadata !{i32 786478, metadata !29, metadata !6, metadata !"sum_array", metadata !"sum_array", metadata !"", i32 9, metadata !12, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32*, i32)* @sum_array, null, null, metadata !1, i32 9} ; [ DW_TAG_subprogram ] [line 9] [def] [sum_array]
!12 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !13, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{metadata !10, metadata !9, metadata !10}
!14 = metadata !{i32 786478, metadata !29, metadata !6, metadata !"main", metadata !"main", metadata !"", i32 18, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !1, i32 18} ; [ DW_TAG_subprogram ] [line 18] [def] [main]
!15 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{metadata !10}
!17 = metadata !{i32 786688, metadata !18, metadata !"main_arr", metadata !6, i32 19, metadata !19, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [main_arr] [line 19]
!18 = metadata !{i32 786443, metadata !29, metadata !14, i32 18, i32 16, i32 4} ; [ DW_TAG_lexical_block ] [/home/timnor01/a64-trunk/build/simple.c]
!19 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 3200, i64 32, i32 0, i32 0, metadata !10, metadata !20, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 3200, align 32, offset 0] [from int]
!20 = metadata !{i32 786465, i64 0, i64 99}       ; [ DW_TAG_subrange_type ] [0, 99]
!22 = metadata !{i32 19, i32 7, metadata !18, null}
!23 = metadata !{i32 786688, metadata !18, metadata !"val", metadata !6, i32 20, metadata !10, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [val] [line 20]
!24 = metadata !{i32 20, i32 7, metadata !18, null}
!25 = metadata !{i32 22, i32 3, metadata !18, null}
!26 = metadata !{i32 23, i32 9, metadata !18, null}
!27 = metadata !{i32 24, i32 3, metadata !18, null}
!28 = metadata !{i32 26, i32 3, metadata !18, null}
!29 = metadata !{metadata !"simple.c", metadata !"/home/timnor01/a64-trunk/build"}
!30 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
