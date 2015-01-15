; Verify the behavior of the IntelJITEventListener.
; RUN: llvm-jitlistener %s | FileCheck %s

; This test was created using the following file:
;
; 1: int foo(int a) {
; 2:   return a;
; 3: }
;

; CHECK: Method load [1]: foo, Size = {{[0-9]+}}
; CHECK:   Line info @ {{[0-9]+}}: simple.c, line 1
; CHECK:   Line info @ {{[0-9]+}}: simple.c, line 2
; CHECK: Method unload [1]

; ModuleID = 'simple.c'

; Function Attrs: nounwind uwtable
define i32 @foo(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !12, metadata !13), !dbg !14
  %0 = load i32* %a.addr, align 4, !dbg !15
  ret i32 %0, !dbg !16
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !{!"0x11\0012\00clang version 3.6.0 (trunk)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [F:\users\akaylor\llvm-s\llvm\test\JitListener/simple.c] [DW_LANG_C99]
!1 = !{!"simple.c", !"F:\5Cusers\5Cakaylor\5Cllvm-s\5Cllvm\5Ctest\5CJitListener"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\000\00256\000\001", !1, !5, !6, null, i32 (i32)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}                               ; [ DW_TAG_file_type ] [F:\users\akaylor\llvm-s\llvm\test\JitListener/simple.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 2}
!11 = !{!"clang version 3.6.0 (trunk)"}
!12 = !{!"0x101\00a\0016777217\000", !4, !5, !8}  ; [ DW_TAG_arg_variable ] [a] [line 1]
!13 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!14 = !MDLocation(line: 1, column: 13, scope: !4)
!15 = !MDLocation(line: 2, column: 10, scope: !4)
!16 = !MDLocation(line: 2, column: 3, scope: !4)
