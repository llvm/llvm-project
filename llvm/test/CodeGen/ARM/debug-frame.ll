; ARM EHABI integrated test

; This test case checks whether the ARM DWARF stack frame directives
; are properly generated or not.

; We have to check several cases:
; (1) arm with -disable-fp-elim
; (2) arm without -disable-fp-elim
; (3) armv7 with -disable-fp-elim
; (4) armv7 without -disable-fp-elim
; (5) thumb with -disable-fp-elim
; (6) thumb without -disable-fp-elim
; (7) thumbv7 with -disable-fp-elim
; (8) thumbv7 without -disable-fp-elim
; (9) thumbv7 with -no-integrated-as

; RUN: llc -mtriple arm-unknown-linux-gnueabi \
; RUN:     -disable-fp-elim -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP

; RUN: llc -mtriple arm-unknown-linux-gnueabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP-ELIM

; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -disable-fp-elim -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-V7-FP

; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-V7-FP-ELIM

; RUN: llc -mtriple thumbv5-unknown-linux-gnueabi \
; RUN:     -disable-fp-elim -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-THUMB-FP

; RUN: llc -mtriple thumbv5-unknown-linux-gnueabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-THUMB-FP-ELIM

; RUN: llc -mtriple thumbv7-unknown-linux-gnueabi \
; RUN:     -disable-fp-elim -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-THUMB-V7-FP

; RUN: llc -mtriple thumbv7-unknown-linux-gnueabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-THUMB-V7-FP-ELIM

; RUN: llc -mtriple thumbv7-unknown-linux-gnueabi \
; RUN:     -disable-fp-elim -no-integrated-as -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-THUMB-V7-FP-NOIAS

;-------------------------------------------------------------------------------
; Test 1
;-------------------------------------------------------------------------------
; This is the LLVM assembly generated from following C++ code:
;
;   extern void print(int, int, int, int, int);
;   extern void print(double, double, double, double, double);
;
;   void test(int a, int b, int c, int d, int e,
;             double m, double n, double p, double q, double r) {
;     try {
;       print(a, b, c, d, e);
;     } catch (...) {
;       print(m, n, p, q, r);
;     }
;   }

declare void @_Z5printiiiii(i32, i32, i32, i32, i32)

declare void @_Z5printddddd(double, double, double, double, double)

define void @_Z4testiiiiiddddd(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e,
                               double %m, double %n, double %p,
                               double %q, double %r) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @_Z5printiiiii(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e)
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = tail call i8* @__cxa_begin_catch(i8* %1)
  invoke void @_Z5printddddd(double %m, double %n, double %p,
                             double %q, double %r)
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret void

lpad1:
  %3 = landingpad { i8*, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad

eh.resume:
  resume { i8*, i32 } %3

terminate.lpad:
  %4 = landingpad { i8*, i32 }
          catch i8* null
  %5 = extractvalue { i8*, i32 } %4, 0
  tail call void @__clang_call_terminate(i8* %5)
  unreachable
}

declare void @__clang_call_terminate(i8*)

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare void @_ZSt9terminatev()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "exp.cpp", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "test", linkageName: "_Z4testiiiiiddddd", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "exp.cpp", directory: "/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8, !8, !8, !8, !9, !9, !9, !9, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 1, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.5 "}
!13 = !DILocalVariable(name: "a", line: 4, arg: 1, scope: !4, file: !5, type: !8)
!14 = !DILocation(line: 4, scope: !4)
!15 = !DILocalVariable(name: "b", line: 4, arg: 2, scope: !4, file: !5, type: !8)
!16 = !DILocalVariable(name: "c", line: 4, arg: 3, scope: !4, file: !5, type: !8)
!17 = !DILocalVariable(name: "d", line: 4, arg: 4, scope: !4, file: !5, type: !8)
!18 = !DILocalVariable(name: "e", line: 4, arg: 5, scope: !4, file: !5, type: !8)
!19 = !DILocalVariable(name: "m", line: 5, arg: 6, scope: !4, file: !5, type: !9)
!20 = !DILocation(line: 5, scope: !4)
!21 = !DILocalVariable(name: "n", line: 5, arg: 7, scope: !4, file: !5, type: !9)
!22 = !DILocalVariable(name: "p", line: 5, arg: 8, scope: !4, file: !5, type: !9)
!23 = !DILocalVariable(name: "q", line: 5, arg: 9, scope: !4, file: !5, type: !9)
!24 = !DILocalVariable(name: "r", line: 5, arg: 10, scope: !4, file: !5, type: !9)
!25 = !DILocation(line: 7, scope: !26)
!26 = distinct !DILexicalBlock(line: 6, column: 0, file: !1, scope: !4)
!27 = !DILocation(line: 8, scope: !26)
!28 = !DILocation(line: 11, scope: !26)
!29 = !DILocation(line: 9, scope: !30)
!30 = distinct !DILexicalBlock(line: 8, column: 0, file: !1, scope: !4)
!31 = !DILocation(line: 10, scope: !30)
!32 = !DILocation(line: 10, scope: !4)
!33 = !DILocation(line: 11, scope: !4)
!34 = !DILocation(line: 11, scope: !30)

; CHECK-FP-LABEL: _Z4testiiiiiddddd:
; CHECK-FP:   .cfi_startproc
; CHECK-FP:   push   {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-FP:   .cfi_def_cfa_offset 36
; CHECK-FP:   .cfi_offset lr, -4
; CHECK-FP:   .cfi_offset r11, -8
; CHECK-FP:   .cfi_offset r10, -12
; CHECK-FP:   .cfi_offset r9, -16
; CHECK-FP:   .cfi_offset r8, -20
; CHECK-FP:   .cfi_offset r7, -24
; CHECK-FP:   .cfi_offset r6, -28
; CHECK-FP:   .cfi_offset r5, -32
; CHECK-FP:   .cfi_offset r4, -36
; CHECK-FP:   add    r11, sp, #28
; CHECK-FP:   .cfi_def_cfa r11, 8
; CHECK-FP:   sub    sp, sp, #44
; CHECK-FP:   .cfi_endproc

; CHECK-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; CHECK-FP-ELIM:   .cfi_startproc
; CHECK-FP-ELIM:   push  {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-FP-ELIM:   .cfi_def_cfa_offset 36
; CHECK-FP-ELIM:   .cfi_offset lr, -4
; CHECK-FP-ELIM:   .cfi_offset r11, -8
; CHECK-FP-ELIM:   .cfi_offset r10, -12
; CHECK-FP-ELIM:   .cfi_offset r9, -16
; CHECK-FP-ELIM:   .cfi_offset r8, -20
; CHECK-FP-ELIM:   .cfi_offset r7, -24
; CHECK-FP-ELIM:   .cfi_offset r6, -28
; CHECK-FP-ELIM:   .cfi_offset r5, -32
; CHECK-FP-ELIM:   .cfi_offset r4, -36
; CHECK-FP-ELIM:   sub   sp, sp, #36
; CHECK-FP-ELIM:   .cfi_def_cfa_offset 72
; CHECK-FP-ELIM:   .cfi_endproc

; CHECK-V7-FP-LABEL: _Z4testiiiiiddddd:
; CHECK-V7-FP:   .cfi_startproc
; CHECK-V7-FP:   push   {r4, r10, r11, lr}
; CHECK-V7-FP:   .cfi_def_cfa_offset 16
; CHECK-V7-FP:   .cfi_offset lr, -4
; CHECK-V7-FP:   .cfi_offset r11, -8
; CHECK-V7-FP:   .cfi_offset r10, -12
; CHECK-V7-FP:   .cfi_offset r4, -16
; CHECK-V7-FP:   add    r11, sp, #8
; CHECK-V7-FP:   .cfi_def_cfa r11, 8
; CHECK-V7-FP:   vpush  {d8, d9, d10, d11, d12}
; CHECK-V7-FP:   .cfi_offset d12, -24
; CHECK-V7-FP:   .cfi_offset d11, -32
; CHECK-V7-FP:   .cfi_offset d10, -40
; CHECK-V7-FP:   .cfi_offset d9, -48
; CHECK-V7-FP:   .cfi_offset d8, -56
; CHECK-V7-FP:   sub    sp, sp, #24
; CHECK-V7-FP:   .cfi_endproc

; CHECK-V7-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; CHECK-V7-FP-ELIM:   .cfi_startproc
; CHECK-V7-FP-ELIM:   push   {r4, lr}
; CHECK-V7-FP-ELIM:   .cfi_def_cfa_offset 8
; CHECK-V7-FP-ELIM:   .cfi_offset lr, -4
; CHECK-V7-FP-ELIM:   .cfi_offset r4, -8
; CHECK-V7-FP-ELIM:   vpush  {d8, d9, d10, d11, d12}
; CHECK-V7-FP-ELIM:   .cfi_def_cfa_offset 48
; CHECK-V7-FP-ELIM:   .cfi_offset d12, -16
; CHECK-V7-FP-ELIM:   .cfi_offset d11, -24
; CHECK-V7-FP-ELIM:   .cfi_offset d10, -32
; CHECK-V7-FP-ELIM:   .cfi_offset d9, -40
; CHECK-V7-FP-ELIM:   .cfi_offset d8, -48
; CHECK-V7-FP-ELIM:   sub    sp, sp, #24
; CHECK-V7-FP-ELIM:   .cfi_def_cfa_offset 72
; CHECK-V7-FP-ELIM:   .cfi_endproc

; CHECK-THUMB-FP-LABEL: _Z4testiiiiiddddd:
; CHECK-THUMB-FP:   .cfi_startproc
; CHECK-THUMB-FP:   push   {r4, r5, r6, r7, lr}
; CHECK-THUMB-FP:   .cfi_def_cfa_offset 20
; CHECK-THUMB-FP:   .cfi_offset lr, -4
; CHECK-THUMB-FP:   .cfi_offset r7, -8
; CHECK-THUMB-FP:   .cfi_offset r6, -12
; CHECK-THUMB-FP:   .cfi_offset r5, -16
; CHECK-THUMB-FP:   .cfi_offset r4, -20
; CHECK-THUMB-FP:   add    r7, sp, #12
; CHECK-THUMB-FP:   .cfi_def_cfa r7, 8
; CHECK-THUMB-FP:   sub    sp, #60
; CHECK-THUMB-FP:   .cfi_endproc

; CHECK-THUMB-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; CHECK-THUMB-FP-ELIM:   .cfi_startproc
; CHECK-THUMB-FP-ELIM:   push   {r4, r5, r6, r7, lr}
; CHECK-THUMB-FP-ELIM:   .cfi_def_cfa_offset 20
; CHECK-THUMB-FP-ELIM:   .cfi_offset lr, -4
; CHECK-THUMB-FP-ELIM:   .cfi_offset r7, -8
; CHECK-THUMB-FP-ELIM:   .cfi_offset r6, -12
; CHECK-THUMB-FP-ELIM:   .cfi_offset r5, -16
; CHECK-THUMB-FP-ELIM:   .cfi_offset r4, -20
; CHECK-THUMB-FP-ELIM:   sub    sp, #60
; CHECK-THUMB-FP-ELIM:   .cfi_def_cfa_offset 80
; CHECK-THUMB-FP-ELIM:   .cfi_endproc

; CHECK-THUMB-V7-FP-LABEL: _Z4testiiiiiddddd:
; CHECK-THUMB-V7-FP:   .cfi_startproc
; CHECK-THUMB-V7-FP:   push   {r4, r6, r7, lr}
; CHECK-THUMB-V7-FP:   .cfi_def_cfa_offset 16
; CHECK-THUMB-V7-FP:   .cfi_offset lr, -4
; CHECK-THUMB-V7-FP:   .cfi_offset r7, -8
; CHECK-THUMB-V7-FP:   .cfi_offset r6, -12
; CHECK-THUMB-V7-FP:   .cfi_offset r4, -16
; CHECK-THUMB-V7-FP:   add    r7, sp, #8
; CHECK-THUMB-V7-FP:   .cfi_def_cfa r7, 8
; CHECK-THUMB-V7-FP:   vpush  {d8, d9, d10, d11, d12}
; CHECK-THUMB-V7-FP:   .cfi_offset d12, -24
; CHECK-THUMB-V7-FP:   .cfi_offset d11, -32
; CHECK-THUMB-V7-FP:   .cfi_offset d10, -40
; CHECK-THUMB-V7-FP:   .cfi_offset d9, -48
; CHECK-THUMB-V7-FP:   .cfi_offset d8, -56
; CHECK-THUMB-V7-FP:   sub    sp, #24
; CHECK-THUMB-V7-FP:   .cfi_endproc

; CHECK-THUMB-V7-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; CHECK-THUMB-V7-FP-ELIM:   .cfi_startproc
; CHECK-THUMB-V7-FP-ELIM:   push   {r4, lr}
; CHECK-THUMB-V7-FP-ELIM:   .cfi_def_cfa_offset 8
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset lr, -4
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset r4, -8
; CHECK-THUMB-V7-FP-ELIM:   vpush  {d8, d9, d10, d11, d12}
; CHECK-THUMB-V7-FP-ELIM:   .cfi_def_cfa_offset 48
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset d12, -16
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset d11, -24
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset d10, -32
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset d9, -40
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset d8, -48
; CHECK-THUMB-V7-FP-ELIM:   sub    sp, #24
; CHECK-THUMB-V7-FP-ELIM:   .cfi_def_cfa_offset 72
; CHECK-THUMB-V7-FP-ELIM:   .cfi_endproc

; CHECK-THUMB-V7-FP-NOIAS-LABEL: _Z4testiiiiiddddd:
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_startproc
; CHECK-THUMB-V7-FP-NOIAS:   push   {r4, r6, r7, lr}
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_def_cfa_offset 16
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_offset 14, -4
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_offset 7, -8
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_offset 6, -12
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_offset 4, -16
; CHECK-THUMB-V7-FP-NOIAS:   add    r7, sp, #8
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_def_cfa 7, 8
; CHECK-THUMB-V7-FP-NOIAS:   vpush  {d8, d9, d10, d11, d12}
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_offset 268, -24
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_offset 267, -32
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_offset 266, -40
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_offset 265, -48
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_offset 264, -56
; CHECK-THUMB-V7-FP-NOIAS:   sub    sp, #24
; CHECK-THUMB-V7-FP-NOIAS:   .cfi_endproc

;-------------------------------------------------------------------------------
; Test 2
;-------------------------------------------------------------------------------

declare void @throw_exception_2()

define void @test2() {
entry:
  call void @throw_exception_2()
  ret void
}

; CHECK-FP-LABEL: test2:
; CHECK-FP:   .cfi_startproc
; CHECK-FP:   push   {r11, lr}
; CHECK-FP:   .cfi_def_cfa_offset 8
; CHECK-FP:   .cfi_offset lr, -4
; CHECK-FP:   .cfi_offset r11, -8
; CHECK-FP:   mov    r11, sp
; CHECK-FP:   .cfi_def_cfa_register r11
; CHECK-FP:   pop    {r11, lr}
; CHECK-FP:   mov    pc, lr
; CHECK-FP:   .cfi_endproc

; CHECK-FP-ELIM-LABEL: test2:
; CHECK-FP-ELIM:   .cfi_startproc
; CHECK-FP-ELIM:   push  {r11, lr}
; CHECK-FP-ELIM:   .cfi_def_cfa_offset 8
; CHECK-FP-ELIM:   .cfi_offset lr, -4
; CHECK-FP-ELIM:   .cfi_offset r11, -8
; CHECK-FP-ELIM:   pop   {r11, lr}
; CHECK-FP-ELIM:   mov   pc, lr
; CHECK-FP-ELIM:   .cfi_endproc

; CHECK-V7-FP-LABEL: test2:
; CHECK-V7-FP:   .cfi_startproc
; CHECK-V7-FP:   push   {r11, lr}
; CHECK-V7-FP:   .cfi_def_cfa_offset 8
; CHECK-V7-FP:   .cfi_offset lr, -4
; CHECK-V7-FP:   .cfi_offset r11, -8
; CHECK-V7-FP:   mov    r11, sp
; CHECK-V7-FP:   .cfi_def_cfa_register r11
; CHECK-V7-FP:   pop    {r11, pc}
; CHECK-V7-FP:   .cfi_endproc

; CHECK-V7-FP-ELIM-LABEL: test2:
; CHECK-V7-FP-ELIM:   .cfi_startproc
; CHECK-V7-FP-ELIM:   push  {r11, lr}
; CHECK-V7-FP-ELIM:   .cfi_def_cfa_offset 8
; CHECK-V7-FP-ELIM:   .cfi_offset lr, -4
; CHECK-V7-FP-ELIM:   .cfi_offset r11, -8
; CHECK-V7-FP-ELIM:   pop   {r11, pc}
; CHECK-V7-FP-ELIM:   .cfi_endproc

; CHECK-THUMB-FP-LABEL: test2:
; CHECK-THUMB-FP:   .cfi_startproc
; CHECK-THUMB-FP:   push   {r7, lr}
; CHECK-THUMB-FP:   .cfi_def_cfa_offset 8
; CHECK-THUMB-FP:   .cfi_offset lr, -4
; CHECK-THUMB-FP:   .cfi_offset r7, -8
; CHECK-THUMB-FP:   add    r7, sp, #0
; CHECK-THUMB-FP:   .cfi_def_cfa_register r7
; CHECK-THUMB-FP:   pop    {r7, pc}
; CHECK-THUMB-FP:   .cfi_endproc

; CHECK-THUMB-FP-ELIM-LABEL: test2:
; CHECK-THUMB-FP-ELIM:   .cfi_startproc
; CHECK-THUMB-FP-ELIM:   push  {r7, lr}
; CHECK-THUMB-FP-ELIM:   .cfi_def_cfa_offset 8
; CHECK-THUMB-FP-ELIM:   .cfi_offset lr, -4
; CHECK-THUMB-FP-ELIM:   .cfi_offset r7, -8
; CHECK-THUMB-FP-ELIM:   pop   {r7, pc}
; CHECK-THUMB-FP-ELIM:   .cfi_endproc

; CHECK-THUMB-V7-FP-LABEL: test2:
; CHECK-THUMB-V7-FP:   .cfi_startproc
; CHECK-THUMB-V7-FP:   push   {r7, lr}
; CHECK-THUMB-V7-FP:   .cfi_def_cfa_offset 8
; CHECK-THUMB-V7-FP:   .cfi_offset lr, -4
; CHECK-THUMB-V7-FP:   .cfi_offset r7, -8
; CHECK-THUMB-V7-FP:   mov    r7, sp
; CHECK-THUMB-V7-FP:   .cfi_def_cfa_register r7
; CHECK-THUMB-V7-FP:   pop    {r7, pc}
; CHECK-THUMB-V7-FP:   .cfi_endproc

; CHECK-THUMB-V7-FP-ELIM-LABEL: test2:
; CHECK-THUMB-V7-FP-ELIM:   .cfi_startproc
; CHECK-THUMB-V7-FP-ELIM:   push  {r7, lr}
; CHECK-THUMB-V7-FP-ELIM:   .cfi_def_cfa_offset 8
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset lr, -4
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset r7, -8
; CHECK-THUMB-V7-FP-ELIM:   pop   {r7, pc}
; CHECK-THUMB-V7-FP-ELIM:   .cfi_endproc


;-------------------------------------------------------------------------------
; Test 3
;-------------------------------------------------------------------------------

declare void @throw_exception_3(i32)

define i32 @test3(i32 %a, i32 %b, i32 %c, i32 %d,
                  i32 %e, i32 %f, i32 %g, i32 %h) {
entry:
  %add = add nsw i32 %b, %a
  %add1 = add nsw i32 %add, %c
  %add2 = add nsw i32 %add1, %d
  tail call void @throw_exception_3(i32 %add2)
  %add3 = add nsw i32 %f, %e
  %add4 = add nsw i32 %add3, %g
  %add5 = add nsw i32 %add4, %h
  tail call void @throw_exception_3(i32 %add5)
  %add6 = add nsw i32 %add5, %add2
  ret i32 %add6
}

; CHECK-FP-LABEL: test3:
; CHECK-FP:   .cfi_startproc
; CHECK-FP:   push   {r4, r5, r11, lr}
; CHECK-FP:   .cfi_def_cfa_offset 16
; CHECK-FP:   .cfi_offset lr, -4
; CHECK-FP:   .cfi_offset r11, -8
; CHECK-FP:   .cfi_offset r5, -12
; CHECK-FP:   .cfi_offset r4, -16
; CHECK-FP:   add    r11, sp, #8
; CHECK-FP:   .cfi_def_cfa r11, 8
; CHECK-FP:   pop    {r4, r5, r11, lr}
; CHECK-FP:   mov    pc, lr
; CHECK-FP:   .cfi_endproc

; CHECK-FP-ELIM-LABEL: test3:
; CHECK-FP-ELIM:   .cfi_startproc
; CHECK-FP-ELIM:   push  {r4, r5, r11, lr}
; CHECK-FP-ELIM:   .cfi_def_cfa_offset 16
; CHECK-FP-ELIM:   .cfi_offset lr, -4
; CHECK-FP-ELIM:   .cfi_offset r11, -8
; CHECK-FP-ELIM:   .cfi_offset r5, -12
; CHECK-FP-ELIM:   .cfi_offset r4, -16
; CHECK-FP-ELIM:   pop   {r4, r5, r11, lr}
; CHECK-FP-ELIM:   mov   pc, lr
; CHECK-FP-ELIM:   .cfi_endproc

; CHECK-V7-FP-LABEL: test3:
; CHECK-V7-FP:   .cfi_startproc
; CHECK-V7-FP:   push   {r4, r5, r11, lr}
; CHECK-V7-FP:   .cfi_def_cfa_offset 16
; CHECK-V7-FP:   .cfi_offset lr, -4
; CHECK-V7-FP:   .cfi_offset r11, -8
; CHECK-V7-FP:   .cfi_offset r5, -12
; CHECK-V7-FP:   .cfi_offset r4, -16
; CHECK-V7-FP:   add    r11, sp, #8
; CHECK-V7-FP:   .cfi_def_cfa r11, 8
; CHECK-V7-FP:   pop    {r4, r5, r11, pc}
; CHECK-V7-FP:   .cfi_endproc

; CHECK-V7-FP-ELIM-LABEL: test3:
; CHECK-V7-FP-ELIM:   .cfi_startproc
; CHECK-V7-FP-ELIM:   push  {r4, r5, r11, lr}
; CHECK-V7-FP-ELIM:   .cfi_def_cfa_offset 16
; CHECK-V7-FP-ELIM:   .cfi_offset lr, -4
; CHECK-V7-FP-ELIM:   .cfi_offset r11, -8
; CHECK-V7-FP-ELIM:   .cfi_offset r5, -12
; CHECK-V7-FP-ELIM:   .cfi_offset r4, -16
; CHECK-V7-FP-ELIM:   pop   {r4, r5, r11, pc}
; CHECK-V7-FP-ELIM:   .cfi_endproc

; CHECK-THUMB-FP-LABEL: test3:
; CHECK-THUMB-FP:   .cfi_startproc
; CHECK-THUMB-FP:   push   {r4, r5, r7, lr}
; CHECK-THUMB-FP:   .cfi_def_cfa_offset 16
; CHECK-THUMB-FP:   .cfi_offset lr, -4
; CHECK-THUMB-FP:   .cfi_offset r7, -8
; CHECK-THUMB-FP:   .cfi_offset r5, -12
; CHECK-THUMB-FP:   .cfi_offset r4, -16
; CHECK-THUMB-FP:   add    r7, sp, #8
; CHECK-THUMB-FP:   .cfi_def_cfa r7, 8
; CHECK-THUMB-FP:   pop    {r4, r5, r7, pc}
; CHECK-THUMB-FP:   .cfi_endproc

; CHECK-THUMB-FP-ELIM-LABEL: test3:
; CHECK-THUMB-FP-ELIM:   .cfi_startproc
; CHECK-THUMB-FP-ELIM:   push  {r4, r5, r7, lr}
; CHECK-THUMB-FP-ELIM:   .cfi_def_cfa_offset 16
; CHECK-THUMB-FP-ELIM:   .cfi_offset lr, -4
; CHECK-THUMB-FP-ELIM:   .cfi_offset r7, -8
; CHECK-THUMB-FP-ELIM:   .cfi_offset r5, -12
; CHECK-THUMB-FP-ELIM:   .cfi_offset r4, -16
; CHECK-THUMB-FP-ELIM:   pop   {r4, r5, r7, pc}
; CHECK-THUMB-FP-ELIM:   .cfi_endproc

; CHECK-THUMB-V7-FP-LABEL: test3:
; CHECK-THUMB-V7-FP:   .cfi_startproc
; CHECK-THUMB-V7-FP:   push   {r4, r5, r7, lr}
; CHECK-THUMB-V7-FP:   .cfi_def_cfa_offset 16
; CHECK-THUMB-V7-FP:   .cfi_offset lr, -4
; CHECK-THUMB-V7-FP:   .cfi_offset r7, -8
; CHECK-THUMB-V7-FP:   .cfi_offset r5, -12
; CHECK-THUMB-V7-FP:   .cfi_offset r4, -16
; CHECK-THUMB-V7-FP:   add    r7, sp, #8
; CHECK-THUMB-V7-FP:   .cfi_def_cfa r7, 8
; CHECK-THUMB-V7-FP:   pop    {r4, r5, r7, pc}
; CHECK-THUMB-V7-FP:   .cfi_endproc

; CHECK-THUMB-V7-FP-ELIM-LABEL: test3:
; CHECK-THUMB-V7-FP-ELIM:   .cfi_startproc
; CHECK-THUMB-V7-FP-ELIM:   push  {r4, r5, r7, lr}
; CHECK-THUMB-V7-FP-ELIM:   .cfi_def_cfa_offset 16
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset lr, -4
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset r7, -8
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset r5, -12
; CHECK-THUMB-V7-FP-ELIM:   .cfi_offset r4, -16
; CHECK-THUMB-V7-FP-ELIM:   pop   {r4, r5, r7, pc}
; CHECK-THUMB-V7-FP-ELIM:   .cfi_endproc


;-------------------------------------------------------------------------------
; Test 4
;-------------------------------------------------------------------------------

define void @test4() nounwind {
entry:
  ret void
}

; CHECK-FP-LABEL: test4:
; CHECK-FP:   mov pc, lr
; CHECK-FP-NOT:   .cfi_def_cfa_offset

; CHECK-FP-ELIM-LABEL: test4:
; CHECK-FP-ELIM:   mov pc, lr
; CHECK-FP-ELIM-NOT:   .cfi_def_cfa_offset

; CHECK-V7-FP-LABEL: test4:
; CHECK-V7-FP:   bx lr
; CHECK-V7-FP-NOT:   .cfi_def_cfa_offset

; CHECK-V7-FP-ELIM-LABEL: test4:
; CHECK-V7-FP-ELIM:   bx lr
; CHECK-V7-FP-ELIM-NOT:   .cfi_def_cfa_offset

; CHECK-THUMB-FP-LABEL: test4:
; CHECK-THUMB-FP:   bx lr
; CHECK-THUMB-FP-NOT:   .cfi_def_cfa_offset

; CHECK-THUMB-FP-ELIM-LABEL: test4:
; CHECK-THUMB-FP-ELIM:   bx lr
; CHECK-THUMB-FP-ELIM-NOT:   .cfi_def_cfa_offset

; CHECK-THUMB-V7-FP-LABEL: test4:
; CHECK-THUMB-V7-FP:   bx lr
; CHECK-THUMB-V7-FP-NOT:   .cfi_def_cfa_offset

; CHECK-THUMB-V7-FP-ELIM-LABEL: test4:
; CHECK-THUMB-V7-FP-ELIM:   bx lr
; CHECK-THUMB-V7-FP-ELIM-NOT:   .cfi_def_cfa_offset

