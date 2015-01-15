; Verify the behavior of the IntelJITEventListener.
; RUN: llvm-jitlistener %s | FileCheck %s

; This test was created using the following file:
;
;  1: int foo(int a) {
;  2:   return a;
;  3: }
;  4:
;  5: int bar(int a) {
;  6:   if (a == 0) {
;  7:     return 0;
;  8:   }
;  9:   return 100/a;
; 10: }
; 11: 
; 12: int fubar(int a) {
; 13:   switch (a) {
; 14:     case 0:
; 15:       return 10;
; 16:     case 1:
; 17:       return 20;
; 18:     default:
; 19:       return 30;
; 20:   }
; 21: }
;

; CHECK: Method load [1]: bar, Size = {{[0-9]+}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[5,6,7,9]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[5,6,7,9]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[5,6,7,9]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[5,6,7,9]}}

; CHECK: Method load [2]: foo, Size = {{[0-9]+}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[1,2]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[1,2]}}

; CHECK: Method load [3]: fubar, Size = {{[0-9]+}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}
; CHECK:   Line info @ {{[0-9]+}}: multiple.c, line {{[12,13,15,17,19]}}

; CHECK: Method unload [1]
; CHECK: Method unload [2]
; CHECK: Method unload [3]

; ModuleID = 'multiple.c'

; Function Attrs: nounwind uwtable
define i32 @foo(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !15, metadata !16), !dbg !17
  %0 = load i32* %a.addr, align 4, !dbg !18
  ret i32 %0, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %a) #0 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !20, metadata !16), !dbg !21
  %0 = load i32* %a.addr, align 4, !dbg !22
  %cmp = icmp eq i32 %0, 0, !dbg !22
  br i1 %cmp, label %if.then, label %if.end, !dbg !24

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, !dbg !25
  br label %return, !dbg !25

if.end:                                           ; preds = %entry
  %1 = load i32* %a.addr, align 4, !dbg !27
  %div = sdiv i32 100, %1, !dbg !28
  store i32 %div, i32* %retval, !dbg !29
  br label %return, !dbg !29

return:                                           ; preds = %if.end, %if.then
  %2 = load i32* %retval, !dbg !30
  ret i32 %2, !dbg !30
}

; Function Attrs: nounwind uwtable
define i32 @fubar(i32 %a) #0 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !31, metadata !16), !dbg !32
  %0 = load i32* %a.addr, align 4, !dbg !33
  switch i32 %0, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
  ], !dbg !34

sw.bb:                                            ; preds = %entry
  store i32 10, i32* %retval, !dbg !35
  br label %return, !dbg !35

sw.bb1:                                           ; preds = %entry
  store i32 20, i32* %retval, !dbg !37
  br label %return, !dbg !37

sw.default:                                       ; preds = %entry
  store i32 30, i32* %retval, !dbg !38
  br label %return, !dbg !38

return:                                           ; preds = %sw.default, %sw.bb1, %sw.bb
  %1 = load i32* %retval, !dbg !39
  ret i32 %1, !dbg !39
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = !{!"0x11\0012\00clang version 3.6.0 (trunk)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [F:\users\akaylor\llvm-s\llvm\test\JitListener/multiple.c] [DW_LANG_C99]
!1 = !{!"multiple.c", !"F:\5Cusers\5Cakaylor\5Cllvm-s\5Cllvm\5Ctest\5CJitListener"}
!2 = !{}
!3 = !{!4, !9, !10}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\000\00256\000\001", !1, !5, !6, null, i32 (i32)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}                               ; [ DW_TAG_file_type ] [F:\users\akaylor\llvm-s\llvm\test\JitListener/multiple.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x2e\00bar\00bar\00\005\000\001\000\000\00256\000\005", !1, !5, !6, null, i32 (i32)* @bar, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [bar]
!10 = !{!"0x2e\00fubar\00fubar\00\0012\000\001\000\000\00256\000\0012", !1, !5, !6, null, i32 (i32)* @fubar, null, null, !2} ; [ DW_TAG_subprogram ] [line 12] [def] [fubar]
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 2}
!13 = !{i32 1, !"PIC Level", i32 2}
!14 = !{!"clang version 3.6.0 (trunk)"}
!15 = !{!"0x101\00a\0016777217\000", !4, !5, !8}  ; [ DW_TAG_arg_variable ] [a] [line 1]
!16 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!17 = !MDLocation(line: 1, column: 13, scope: !4)
!18 = !MDLocation(line: 2, column: 10, scope: !4)
!19 = !MDLocation(line: 2, column: 3, scope: !4)
!20 = !{!"0x101\00a\0016777221\000", !9, !5, !8}  ; [ DW_TAG_arg_variable ] [a] [line 5]
!21 = !MDLocation(line: 5, column: 13, scope: !9)
!22 = !MDLocation(line: 6, column: 7, scope: !23)
!23 = !{!"0xb\006\007\000", !1, !9}               ; [ DW_TAG_lexical_block ] [F:\users\akaylor\llvm-s\llvm\test\JitListener/multiple.c]
!24 = !MDLocation(line: 6, column: 7, scope: !9)
!25 = !MDLocation(line: 7, column: 5, scope: !26)
!26 = !{!"0xb\006\0015\001", !1, !23}             ; [ DW_TAG_lexical_block ] [F:\users\akaylor\llvm-s\llvm\test\JitListener/multiple.c]
!27 = !MDLocation(line: 9, column: 14, scope: !9)
!28 = !MDLocation(line: 9, column: 10, scope: !9)
!29 = !MDLocation(line: 9, column: 3, scope: !9)
!30 = !MDLocation(line: 10, column: 1, scope: !9)
!31 = !{!"0x101\00a\0016777228\000", !10, !5, !8} ; [ DW_TAG_arg_variable ] [a] [line 12]
!32 = !MDLocation(line: 12, column: 15, scope: !10)
!33 = !MDLocation(line: 13, column: 11, scope: !10)
!34 = !MDLocation(line: 13, column: 3, scope: !10)
!35 = !MDLocation(line: 15, column: 7, scope: !36)
!36 = !{!"0xb\0013\0014\002", !1, !10}            ; [ DW_TAG_lexical_block ] [F:\users\akaylor\llvm-s\llvm\test\JitListener/multiple.c]
!37 = !MDLocation(line: 17, column: 7, scope: !36)
!38 = !MDLocation(line: 19, column: 7, scope: !36)
!39 = !MDLocation(line: 21, column: 1, scope: !10)
