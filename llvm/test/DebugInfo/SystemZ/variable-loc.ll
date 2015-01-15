; RUN: llc -mtriple=s390x-linux-gnu -disable-fp-elim < %s | FileCheck %s
; RUN: llc -mtriple=s390x-linux-gnu -disable-fp-elim -filetype=obj < %s \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=DEBUG %s
;
; This is a regression test making sure the location of variables is correct in
; debugging information, even if they're addressed via the frame pointer.
; Originally a copy of the AArch64 test, commandeered for SystemZ.
;
; First make sure main_arr is where we expect it: %r11 + 164
;
; CHECK: main:
; CHECK: aghi    %r15, -568
; CHECK: la      %r2, 164(%r11)
; CHECK: brasl   %r14, populate_array@PLT

; DEBUG: DW_TAG_variable
; Rather hard-coded, but 0x91 => DW_OP_fbreg and 0xa401 is SLEB128 encoded 164.
; DEBUG-NOT: DW_TAG
; DEBUG: DW_AT_location {{.*}}(<0x3> 91 a4 01 )
; DEBUG-NOT: DW_TAG
; DEBUG: DW_AT_name {{.*}} "main_arr"


@.str = private unnamed_addr constant [13 x i8] c"Total is %d\0A\00", align 2

declare void @populate_array(i32*, i32) nounwind

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @sum_array(i32*, i32) nounwind

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  %main_arr = alloca [100 x i32], align 4
  %val = alloca i32, align 4
  store volatile i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata [100 x i32]* %main_arr, metadata !17, metadata !{!"0x102"}), !dbg !22
  call void @llvm.dbg.declare(metadata i32* %val, metadata !23, metadata !{!"0x102"}), !dbg !24
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

!0 = !{!"0x11\0012\00clang version 3.2 \000\00\000\00\000", !29, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ] [/home/timnor01/a64-trunk/build/simple.c] [DW_LANG_C99]
!1 = !{}
!3 = !{!5, !11, !14}
!5 = !{!"0x2e\00populate_array\00populate_array\00\004\000\001\000\006\00256\000\004", !29, !6, !7, null, void (i32*, i32)* @populate_array, null, null, !1} ; [ DW_TAG_subprogram ] [line 4] [def] [populate_array]
!6 = !{!"0x29", !29} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null, !9, !10}
!9 = !{!"0xf\00\000\0064\0064\000\000", null, null, !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!10 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = !{!"0x2e\00sum_array\00sum_array\00\009\000\001\000\006\00256\000\009", !29, !6, !12, null, i32 (i32*, i32)* @sum_array, null, null, !1} ; [ DW_TAG_subprogram ] [line 9] [def] [sum_array]
!12 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = !{!10, !9, !10}
!14 = !{!"0x2e\00main\00main\00\0018\000\001\000\006\00256\000\0018", !29, !6, !15, null, i32 ()* @main, null, null, !1} ; [ DW_TAG_subprogram ] [line 18] [def] [main]
!15 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{!10}
!17 = !{!"0x100\00main_arr\0019\000", !18, !6, !19} ; [ DW_TAG_auto_variable ] [main_arr] [line 19]
!18 = !{!"0xb\0018\0016\004", !29, !14} ; [ DW_TAG_lexical_block ] [/home/timnor01/a64-trunk/build/simple.c]
!19 = !{!"0x1\00\000\003200\0032\000\000", null, null, !10, !20, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 3200, align 32, offset 0] [from int]
!20 = !{!"0x21\000\0099"}       ; [ DW_TAG_subrange_type ] [0, 99]
!22 = !MDLocation(line: 19, column: 7, scope: !18)
!23 = !{!"0x100\00val\0020\000", !18, !6, !10} ; [ DW_TAG_auto_variable ] [val] [line 20]
!24 = !MDLocation(line: 20, column: 7, scope: !18)
!25 = !MDLocation(line: 22, column: 3, scope: !18)
!26 = !MDLocation(line: 23, column: 9, scope: !18)
!27 = !MDLocation(line: 24, column: 3, scope: !18)
!28 = !MDLocation(line: 26, column: 3, scope: !18)
!29 = !{!"simple.c", !"/home/timnor01/a64-trunk/build"}
!30 = !{i32 1, !"Debug Info Version", i32 2}
