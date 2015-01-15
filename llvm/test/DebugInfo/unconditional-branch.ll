; REQUIRES: object-emission
; PR 19261

; RUN: %llc_dwarf -fast-isel=false -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK: {{0x[0-9a-f]+}}      1      0      1   0             0  is_stmt
; CHECK: {{0x[0-9a-f]+}}      2      0      1   0             0  is_stmt
; CHECK: {{0x[0-9a-f]+}}      4      0      1   0             0  is_stmt

; IR generated from clang -O0 -g with the following source:
;void foo(int i){
;  switch(i){
;  default:
;    break;
;  }
;  return;
;}

; Function Attrs: nounwind
define void @foo(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !12, metadata !{!"0x102"}), !dbg !13
  %0 = load i32* %i.addr, align 4, !dbg !14
  switch i32 %0, label %sw.default [
  ], !dbg !14

sw.epilog:                                        ; preds = %sw.default
  ret void, !dbg !17

sw.default:                                       ; preds = %entry
  br label %sw.epilog, !dbg !15

}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !{!"0x11\0012\00clang version 3.5.0 (204712)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [D:\work\EPRs\396363/test.c] [DW_LANG_C99]
!1 = !{!"test.c", !"D:\5Cwork\5CEPRs\5C396363"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, void (i32)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [D:\work\EPRs\396363/test.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 1, !"Debug Info Version", i32 2}
!11 = !{!"clang version 3.5.0 (204712)"}
!12 = !{!"0x101\00i\0016777217\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [i] [line 1]
!13 = !MDLocation(line: 1, scope: !4)
!14 = !MDLocation(line: 2, scope: !4)
!15 = !MDLocation(line: 4, scope: !16)
!16 = !{!"0xb\002\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [D:\work\EPRs\396363/test.c]
!17 = !MDLocation(line: 6, scope: !4)
