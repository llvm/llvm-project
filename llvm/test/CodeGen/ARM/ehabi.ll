; ARM EHABI integrated test

; This test case checks whether the ARM unwind directives are properly
; generated or not.

; The purpose of the test:
; (1) .fnstart and .fnend directives should wrap the function.
; (2) .setfp directive should be available if frame pointer is not eliminated.
; (3) .save directive should come with push instruction.
; (4) .vsave directive should come with vpush instruction.
; (5) .pad directive should come with stack pointer adjustment.
; (6) .cantunwind directive should be available if the function is marked with
;     nounwind function attribute.

; We have to check several cases:
; (1) arm with -frame-pointer=all
; (2) arm without -frame-pointer=all
; (3) armv7 with -frame-pointer=all
; (4) armv7 without -frame-pointer=all

; RUN: llc -mtriple arm-unknown-linux-gnueabi \
; RUN:     -frame-pointer=all -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP

; RUN: llc -mtriple arm-unknown-linux-gnueabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP-ELIM

; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -frame-pointer=all -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-V7-FP

; RUN: llc -mtriple armv7-unknown-linux-gnueabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-V7-FP-ELIM

; RUN: llc -mtriple arm-unknown-linux-musleabi \
; RUN:     -frame-pointer=all -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP

; RUN: llc -mtriple arm-unknown-linux-musleabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP-ELIM

; RUN: llc -mtriple armv7-unknown-linux-musleabi \
; RUN:     -frame-pointer=all -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-V7-FP

; RUN: llc -mtriple armv7-unknown-linux-musleabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-V7-FP-ELIM

; RUN: llc -mtriple arm-unknown-linux-androideabi \
; RUN:     -frame-pointer=all -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP

; RUN: llc -mtriple arm-unknown-linux-androideabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP-ELIM

; RUN: llc -mtriple armv7-unknown-linux-androideabi \
; RUN:     -frame-pointer=all -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-V7-FP

; RUN: llc -mtriple armv7-unknown-linux-androideabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-V7-FP-ELIM

; RUN: llc -mtriple arm-unknown-netbsd-eabi \
; RUN:     -frame-pointer=all -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=DWARF-FP

; RUN: llc -mtriple arm-unknown-netbsd-eabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=DWARF-FP-ELIM

; RUN: llc -mtriple armv7-unknown-netbsd-eabi \
; RUN:     -frame-pointer=all -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=DWARF-V7-FP

; RUN: llc -mtriple armv7-unknown-netbsd-eabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=DWARF-V7-FP-ELIM

; RUN: llc -mtriple thumbv7-windows-gnu \
; RUN:     -filetype=asm -exception-model=dwarf -o - %s \
; RUN:   | FileCheck %s --check-prefix=DWARF-WIN-FP-ELIM

; RUN: llc -mtriple thumbv7-windows-gnu \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=SEH-WIN-FP-ELIM

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

; CHECK-FP-LABEL: _Z4testiiiiiddddd:
; CHECK-FP:   .fnstart
; CHECK-FP:   .save  {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-FP:   push   {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-FP:   .setfp r11, sp, #28
; CHECK-FP:   add    r11, sp, #28
; CHECK-FP:   .pad   #44
; CHECK-FP:   sub    sp, sp, #44
; CHECK-FP:   .personality __gxx_personality_v0
; CHECK-FP:   .handlerdata
; CHECK-FP:   .fnend

; CHECK-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; CHECK-FP-ELIM:   .fnstart
; CHECK-FP-ELIM:   .save {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-FP-ELIM:   push  {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-FP-ELIM:   .pad  #36
; CHECK-FP-ELIM:   sub   sp, sp, #36
; CHECK-FP-ELIM:   .personality __gxx_personality_v0
; CHECK-FP-ELIM:   .handlerdata
; CHECK-FP-ELIM:   .fnend

; CHECK-V7-FP-LABEL: _Z4testiiiiiddddd:
; CHECK-V7-FP:   .fnstart
; CHECK-V7-FP:   .save  {r11, lr}
; CHECK-V7-FP:   push   {r11, lr}
; CHECK-V7-FP:   .setfp r11, sp
; CHECK-V7-FP:   mov    r11, sp
; CHECK-V7-FP:   .vsave {d8, d9, d10, d11, d12}
; CHECK-V7-FP:   vpush  {d8, d9, d10, d11, d12}
; CHECK-V7-FP:   .pad   #24
; CHECK-V7-FP:   sub    sp, sp, #24
; CHECK-V7-FP:   .personality __gxx_personality_v0
; CHECK-V7-FP:   .handlerdata
; CHECK-V7-FP:   .fnend

; CHECK-V7-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; CHECK-V7-FP-ELIM:   .fnstart
; CHECK-V7-FP-ELIM:   .save  {r11, lr}
; CHECK-V7-FP-ELIM:   push   {r11, lr}
; CHECK-V7-FP-ELIM:   .vsave {d8, d9, d10, d11, d12}
; CHECK-V7-FP-ELIM:   vpush  {d8, d9, d10, d11, d12}
; CHECK-V7-FP-ELIM:   .pad   #24
; CHECK-V7-FP-ELIM:   sub    sp, sp, #24
; CHECK-V7-FP-ELIM:   .personality __gxx_personality_v0
; CHECK-V7-FP-ELIM:   .handlerdata
; CHECK-V7-FP-ELIM:   .fnend

; DWARF-FP-LABEL: _Z4testiiiiiddddd:
; DWARF-FP:    .cfi_startproc
; DWARF-FP:    .cfi_personality 0, __gxx_personality_v0
; DWARF-FP:    .cfi_lsda 0, .Lexception0
; DWARF-FP:    push {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; DWARF-FP:    .cfi_def_cfa_offset 36
; DWARF-FP:    .cfi_offset lr, -4
; DWARF-FP:    .cfi_offset r11, -8
; DWARF-FP:    .cfi_offset r10, -12
; DWARF-FP:    .cfi_offset r9, -16
; DWARF-FP:    .cfi_offset r8, -20
; DWARF-FP:    .cfi_offset r7, -24
; DWARF-FP:    .cfi_offset r6, -28
; DWARF-FP:    .cfi_offset r5, -32
; DWARF-FP:    .cfi_offset r4, -36
; DWARF-FP:    add r11, sp, #28
; DWARF-FP:    .cfi_def_cfa r11, 8
; DWARF-FP:    sub sp, sp, #44
; DWARF-FP:    sub sp, r11, #28
; DWARF-FP:    pop {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; DWARF-FP:    mov pc, lr
; DWARF-FP:    .cfi_endproc

; DWARF-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; DWARF-FP-ELIM:    .cfi_startproc
; DWARF-FP-ELIM:    .cfi_personality 0, __gxx_personality_v0
; DWARF-FP-ELIM:    .cfi_lsda 0, .Lexception0
; DWARF-FP-ELIM:    push {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; DWARF-FP-ELIM:    .cfi_def_cfa_offset 36
; DWARF-FP-ELIM:    .cfi_offset lr, -4
; DWARF-FP-ELIM:    .cfi_offset r11, -8
; DWARF-FP-ELIM:    .cfi_offset r10, -12
; DWARF-FP-ELIM:    .cfi_offset r9, -16
; DWARF-FP-ELIM:    .cfi_offset r8, -20
; DWARF-FP-ELIM:    .cfi_offset r7, -24
; DWARF-FP-ELIM:    .cfi_offset r6, -28
; DWARF-FP-ELIM:    .cfi_offset r5, -32
; DWARF-FP-ELIM:    .cfi_offset r4, -36
; DWARF-FP-ELIM:    sub sp, sp, #36
; DWARF-FP-ELIM:    .cfi_def_cfa_offset 72
; DWARF-FP-ELIM:    add sp, sp, #36
; DWARF-FP-ELIM:    pop {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; DWARF-FP-ELIM:    mov pc, lr
; DWARF-FP-ELIM:    .cfi_endproc

; DWARF-V7-FP-LABEL: _Z4testiiiiiddddd:
; DWARF-V7-FP:    .cfi_startproc
; DWARF-V7-FP:    .cfi_personality 0, __gxx_personality_v0
; DWARF-V7-FP:    .cfi_lsda 0, .Lexception0
; DWARF-V7-FP:    push {r11, lr}
; DWARF-V7-FP:    .cfi_def_cfa_offset 8
; DWARF-V7-FP:    .cfi_offset lr, -4
; DWARF-V7-FP:    .cfi_offset r11, -8
; DWARF-V7-FP:    mov r11, sp
; DWARF-V7-FP:    .cfi_def_cfa_register r11
; DWARF-V7-FP:    vpush {d8, d9, d10, d11, d12}
; DWARF-V7-FP:    .cfi_offset d12, -16
; DWARF-V7-FP:    .cfi_offset d11, -24
; DWARF-V7-FP:    .cfi_offset d10, -32
; DWARF-V7-FP:    .cfi_offset d9, -40
; DWARF-V7-FP:    sub sp, sp, #24
; DWARF-V7-FP:    sub sp, r11, #40
; DWARF-V7-FP:    vpop {d8, d9, d10, d11, d12}
; DWARF-V7-FP:    pop {r11, pc}
; DWARF-V7-FP:    .cfi_endproc

; DWARF-V7-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; DWARF-V7-FP-ELIM:    .cfi_startproc
; DWARF-V7-FP-ELIM:    .cfi_personality 0, __gxx_personality_v0
; DWARF-V7-FP-ELIM:    .cfi_lsda 0, .Lexception0
; DWARF-V7-FP-ELIM:    push {r11, lr}
; DWARF-V7-FP-ELIM:    .cfi_def_cfa_offset 8
; DWARF-V7-FP-ELIM:    .cfi_offset lr, -4
; DWARF-V7-FP-ELIM:    .cfi_offset r11, -8
; DWARF-V7-FP-ELIM:    vpush {d8, d9, d10, d11, d12}
; DWARF-V7-FP-ELIM:    .cfi_offset d12, -16
; DWARF-V7-FP-ELIM:    .cfi_offset d11, -24
; DWARF-V7-FP-ELIM:    .cfi_offset d10, -32
; DWARF-V7-FP-ELIM:    .cfi_offset d9, -40
; DWARF-V7-FP-ELIM:    sub sp, sp, #24
; DWARF-V7-FP-ELIM:    .cfi_def_cfa_offset 72
; DWARF-V7-FP-ELIM:    add sp, sp, #24
; DWARF-V7-FP-ELIM:    vpop {d8, d9, d10, d11, d12}
; DWARF-V7-FP-ELIM:    pop {r11, pc}
; DWARF-V7-FP-ELIM:    .cfi_endproc

; DWARF-WIN-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; DWARF-WIN-FP-ELIM:    .cfi_startproc
; DWARF-WIN-FP-ELIM:    .cfi_personality 0, __gxx_personality_v0
; DWARF-WIN-FP-ELIM:    .cfi_lsda 0, .Lexception0
; DWARF-WIN-FP-ELIM:    push {r4, lr}
; DWARF-WIN-FP-ELIM:    .cfi_def_cfa_offset 8
; DWARF-WIN-FP-ELIM:    .cfi_offset lr, -4
; DWARF-WIN-FP-ELIM:    .cfi_offset r4, -8
; DWARF-WIN-FP-ELIM:    vpush {d8, d9, d10, d11, d12}
; DWARF-WIN-FP-ELIM:    .cfi_offset d12, -16
; DWARF-WIN-FP-ELIM:    .cfi_offset d11, -24
; DWARF-WIN-FP-ELIM:    .cfi_offset d10, -32
; DWARF-WIN-FP-ELIM:    .cfi_offset d9, -40
; DWARF-WIN-FP-ELIM:    sub sp, #8
; DWARF-WIN-FP-ELIM:    .cfi_def_cfa_offset 56
; DWARF-WIN-FP-ELIM:    add sp, #8
; DWARF-WIN-FP-ELIM:    vpop {d8, d9, d10, d11, d12}
; DWARF-WIN-FP-ELIM:    pop {r4, pc}
; DWARF-WIN-FP-ELIM:    .cfi_endproc

; SEH-WIN-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; SEH-WIN-FP-ELIM:    .seh_proc _Z4testiiiiiddddd
; SEH-WIN-FP-ELIM:    .seh_handler __gxx_personality_v0, %unwind, %except
; SEH-WIN-FP-ELIM:    push {r4, lr}
; SEH-WIN-FP-ELIM:    .seh_save_regs {r4, lr}
; SEH-WIN-FP-ELIM:    vpush {d8, d9, d10, d11, d12}
; SEH-WIN-FP-ELIM:    .seh_save_fregs {d8-d12}
; SEH-WIN-FP-ELIM:    sub sp, #8
; SEH-WIN-FP-ELIM:    .seh_stackalloc 8
; SEH-WIN-FP-ELIM:    .seh_endprologue
; SEH-WIN-FP-ELIM:    .seh_startepilogue
; SEH-WIN-FP-ELIM:    add sp, #8
; SEH-WIN-FP-ELIM:    .seh_stackalloc 8
; SEH-WIN-FP-ELIM:    vpop {d8, d9, d10, d11, d12}
; SEH-WIN-FP-ELIM:    .seh_save_fregs {d8-d12}
; SEH-WIN-FP-ELIM:    pop {r4, pc}
; SEH-WIN-FP-ELIM:    .seh_save_regs {r4, lr}
; SEH-WIN-FP-ELIM:    .seh_endepilogue
; SEH-WIN-FP-ELIM:    .seh_endproc

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
; CHECK-FP:   .fnstart
; CHECK-FP:   .save  {r11, lr}
; CHECK-FP:   push   {r11, lr}
; CHECK-FP:   .setfp r11, sp
; CHECK-FP:   mov    r11, sp
; CHECK-FP:   pop    {r11, lr}
; CHECK-FP:   mov    pc, lr
; CHECK-FP:   .fnend

; CHECK-FP-ELIM-LABEL: test2:
; CHECK-FP-ELIM:   .fnstart
; CHECK-FP-ELIM:   .save {r11, lr}
; CHECK-FP-ELIM:   push  {r11, lr}
; CHECK-FP-ELIM:   pop   {r11, lr}
; CHECK-FP-ELIM:   mov   pc, lr
; CHECK-FP-ELIM:   .fnend

; CHECK-V7-FP-LABEL: test2:
; CHECK-V7-FP:   .fnstart
; CHECK-V7-FP:   .save  {r11, lr}
; CHECK-V7-FP:   push   {r11, lr}
; CHECK-V7-FP:   .setfp r11, sp
; CHECK-V7-FP:   mov    r11, sp
; CHECK-V7-FP:   pop    {r11, pc}
; CHECK-V7-FP:   .fnend

; CHECK-V7-FP-ELIM-LABEL: test2:
; CHECK-V7-FP-ELIM:   .fnstart
; CHECK-V7-FP-ELIM:   .save {r11, lr}
; CHECK-V7-FP-ELIM:   push  {r11, lr}
; CHECK-V7-FP-ELIM:   pop   {r11, pc}
; CHECK-V7-FP-ELIM:   .fnend

; DWARF-FP-LABEL: test2:
; DWARF-FP:    .cfi_startproc
; DWARF-FP:    push {r11, lr}
; DWARF-FP:    .cfi_def_cfa_offset 8
; DWARF-FP:    .cfi_offset lr, -4
; DWARF-FP:    .cfi_offset r11, -8
; DWARF-FP:    mov  r11, sp
; DWARF-FP:    .cfi_def_cfa_register r11
; DWARF-FP:    pop  {r11, lr}
; DWARF-FP:    mov  pc, lr
; DWARF-FP:    .cfi_endproc

; DWARF-FP-ELIM-LABEL: test2:
; DWARF-FP-ELIM:    .cfi_startproc
; DWARF-FP-ELIM:    push {r11, lr}
; DWARF-FP-ELIM:    .cfi_def_cfa_offset 8
; DWARF-FP-ELIM:    .cfi_offset lr, -4
; DWARF-FP-ELIM:    .cfi_offset r11, -8
; DWARF-FP-ELIM:    pop  {r11, lr}
; DWARF-FP-ELIM:    mov  pc, lr
; DWARF-FP-ELIM:    .cfi_endproc

; DWARF-V7-FP-LABEL: test2:
; DWARF-V7-FP:    .cfi_startproc
; DWARF-V7-FP:    push {r11, lr}
; DWARF-V7-FP:    .cfi_def_cfa_offset 8
; DWARF-V7-FP:    .cfi_offset lr, -4
; DWARF-V7-FP:    .cfi_offset r11, -8
; DWARF-V7-FP:    mov  r11, sp
; DWARF-V7-FP:    .cfi_def_cfa_register r11
; DWARF-V7-FP:    pop  {r11, pc}
; DWARF-V7-FP:    .cfi_endproc

; DWARF-V7-FP-ELIM-LABEL: test2:
; DWARF-V7-FP-ELIM:    .cfi_startproc
; DWARF-V7-FP-ELIM:    push {r11, lr}
; DWARF-V7-FP-ELIM:    .cfi_def_cfa_offset 8
; DWARF-V7-FP-ELIM:    .cfi_offset lr, -4
; DWARF-V7-FP-ELIM:    .cfi_offset r11, -8
; DWARF-V7-FP-ELIM:    pop  {r11, pc}
; DWARF-V7-FP-ELIM:    .cfi_endproc

; DWARF-WIN-FP-ELIM-LABEL: test2:
; DWARF-WIN-FP-ELIM:    .cfi_startproc
; DWARF-WIN-FP-ELIM:    push.w {r11, lr}
; DWARF-WIN-FP-ELIM:    .cfi_def_cfa_offset 8
; DWARF-WIN-FP-ELIM:    .cfi_offset lr, -4
; DWARF-WIN-FP-ELIM:    .cfi_offset r11, -8
; DWARF-WIN-FP-ELIM:    pop.w  {r11, pc}
; DWARF-WIN-FP-ELIM:    .cfi_endproc

; SEH-WIN-FP-ELIM-LABEL: test2:
; SEH-WIN-FP-ELIM:    .seh_proc test2
; SEH-WIN-FP-ELIM:    push.w {r11, lr}
; SEH-WIN-FP-ELIM:    .seh_save_regs_w {r11, lr}
; SEH-WIN-FP-ELIM:    .seh_endprologue
; SEH-WIN-FP-ELIM:    .seh_startepilogue
; SEH-WIN-FP-ELIM:    pop.w  {r11, pc}
; SEH-WIN-FP-ELIM:    .seh_endepilogue
; SEH-WIN-FP-ELIM:    .seh_endproc


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
; CHECK-FP:   .fnstart
; CHECK-FP:   .save  {r4, r5, r11, lr}
; CHECK-FP:   push   {r4, r5, r11, lr}
; CHECK-FP:   .setfp r11, sp, #8
; CHECK-FP:   add    r11, sp, #8
; CHECK-FP:   pop    {r4, r5, r11, lr}
; CHECK-FP:   mov    pc, lr
; CHECK-FP:   .fnend

; CHECK-FP-ELIM-LABEL: test3:
; CHECK-FP-ELIM:   .fnstart
; CHECK-FP-ELIM:   .save {r4, r5, r11, lr}
; CHECK-FP-ELIM:   push  {r4, r5, r11, lr}
; CHECK-FP-ELIM:   pop   {r4, r5, r11, lr}
; CHECK-FP-ELIM:   mov   pc, lr
; CHECK-FP-ELIM:   .fnend

; CHECK-V7-FP-LABEL: test3:
; CHECK-V7-FP:   .fnstart
; CHECK-V7-FP:   .save  {r4, r5, r11, lr}
; CHECK-V7-FP:   push   {r4, r5, r11, lr}
; CHECK-V7-FP:   .setfp r11, sp, #8
; CHECK-V7-FP:   add    r11, sp, #8
; CHECK-V7-FP:   pop    {r4, r5, r11, pc}
; CHECK-V7-FP:   .fnend

; CHECK-V7-FP-ELIM-LABEL: test3:
; CHECK-V7-FP-ELIM:   .fnstart
; CHECK-V7-FP-ELIM:   .save {r4, r5, r11, lr}
; CHECK-V7-FP-ELIM:   push  {r4, r5, r11, lr}
; CHECK-V7-FP-ELIM:   pop   {r4, r5, r11, pc}
; CHECK-V7-FP-ELIM:   .fnend

; DWARF-FP-LABEL: test3:
; DWARF-FP:    .cfi_startproc
; DWARF-FP:    push {r4, r5, r11, lr}
; DWARF-FP:    .cfi_def_cfa_offset 16
; DWARF-FP:    .cfi_offset lr, -4
; DWARF-FP:    .cfi_offset r11, -8
; DWARF-FP:    .cfi_offset r5, -12
; DWARF-FP:    .cfi_offset r4, -16
; DWARF-FP:    add  r11, sp, #8
; DWARF-FP:    .cfi_def_cfa r11, 8
; DWARF-FP:    pop  {r4, r5, r11, lr}
; DWARF-FP:    mov  pc, lr
; DWARF-FP:    .cfi_endproc

; DWARF-FP-ELIM-LABEL: test3:
; DWARF-FP-ELIM:    .cfi_startproc
; DWARF-FP-ELIM:    push {r4, r5, r11, lr}
; DWARF-FP-ELIM:    .cfi_def_cfa_offset 16
; DWARF-FP-ELIM:    .cfi_offset lr, -4
; DWARF-FP-ELIM:    .cfi_offset r11, -8
; DWARF-FP-ELIM:    .cfi_offset r5, -12
; DWARF-FP-ELIM:    .cfi_offset r4, -16
; DWARF-FP-ELIM:    pop  {r4, r5, r11, lr}
; DWARF-FP-ELIM:    mov  pc, lr
; DWARF-FP-ELIM:    .cfi_endproc

; DWARF-V7-FP-LABEL: test3:
; DWARF-V7-FP:    .cfi_startproc
; DWARF-V7-FP:    push {r4, r5, r11, lr}
; DWARF-V7-FP:    .cfi_def_cfa_offset 16
; DWARF-V7-FP:    .cfi_offset lr, -4
; DWARF-V7-FP:    .cfi_offset r11, -8
; DWARF-V7-FP:    .cfi_offset r5, -12
; DWARF-V7-FP:    .cfi_offset r4, -16
; DWARF-V7-FP:    add  r11, sp, #8
; DWARF-V7-FP:    .cfi_def_cfa r11, 8
; DWARF-V7-FP:    pop  {r4, r5, r11, pc}
; DWARF-V7-FP:    .cfi_endproc

; DWARF-V7-FP-ELIM-LABEL: test3:
; DWARF-V7-FP-ELIM:    .cfi_startproc
; DWARF-V7-FP-ELIM:    push {r4, r5, r11, lr}
; DWARF-V7-FP-ELIM:    .cfi_def_cfa_offset 16
; DWARF-V7-FP-ELIM:    .cfi_offset lr, -4
; DWARF-V7-FP-ELIM:    .cfi_offset r11, -8
; DWARF-V7-FP-ELIM:    .cfi_offset r5, -12
; DWARF-V7-FP-ELIM:    .cfi_offset r4, -16
; DWARF-V7-FP-ELIM:    pop  {r4, r5, r11, pc}
; DWARF-V7-FP-ELIM:    .cfi_endproc

; DWARF-WIN-FP-ELIM-LABEL: test3:
; DWARF-WIN-FP-ELIM:    .cfi_startproc
; DWARF-WIN-FP-ELIM:    push.w {r4, r5, r11, lr}
; DWARF-WIN-FP-ELIM:    .cfi_def_cfa_offset 16
; DWARF-WIN-FP-ELIM:    .cfi_offset lr, -4
; DWARF-WIN-FP-ELIM:    .cfi_offset r11, -8
; DWARF-WIN-FP-ELIM:    .cfi_offset r5, -12
; DWARF-WIN-FP-ELIM:    .cfi_offset r4, -16
; DWARF-WIN-FP-ELIM:    pop.w  {r4, r5, r11, pc}
; DWARF-WIN-FP-ELIM:    .cfi_endproc

; SEH-WIN-FP-ELIM-LABEL: test3:
; SEH-WIN-FP-ELIM:    .seh_proc test3
; SEH-WIN-FP-ELIM:    push.w {r4, r5, r11, lr}
; SEH-WIN-FP-ELIM:    .seh_save_regs_w {r4-r5, r11, lr}
; SEH-WIN-FP-ELIM:    .seh_endprologue
; SEH-WIN-FP-ELIM:    .seh_startepilogue
; SEH-WIN-FP-ELIM:    pop.w  {r4, r5, r11, pc}
; SEH-WIN-FP-ELIM:    .seh_save_regs_w {r4-r5, r11, lr}
; SEH-WIN-FP-ELIM:    .seh_endepilogue
; SEH-WIN-FP-ELIM:    .seh_endproc


;-------------------------------------------------------------------------------
; Test 4
;-------------------------------------------------------------------------------

define void @test4() nounwind {
entry:
  ret void
}

; CHECK-FP-LABEL: test4:
; CHECK-FP:   .fnstart
; CHECK-FP:   mov pc, lr
; CHECK-FP:   .cantunwind
; CHECK-FP:   .fnend

; CHECK-FP-ELIM-LABEL: test4:
; CHECK-FP-ELIM:   .fnstart
; CHECK-FP-ELIM:   mov pc, lr
; CHECK-FP-ELIM:   .cantunwind
; CHECK-FP-ELIM:   .fnend

; CHECK-V7-FP-LABEL: test4:
; CHECK-V7-FP:   .fnstart
; CHECK-V7-FP:   bx lr
; CHECK-V7-FP:   .cantunwind
; CHECK-V7-FP:   .fnend

; CHECK-V7-FP-ELIM-LABEL: test4:
; CHECK-V7-FP-ELIM:   .fnstart
; CHECK-V7-FP-ELIM:   bx lr
; CHECK-V7-FP-ELIM:   .cantunwind
; CHECK-V7-FP-ELIM:   .fnend

; DWARF-FP-LABEL: test4:
; DWARF-FP-NOT: .cfi_startproc
; DWARF-FP:    mov pc, lr
; DWARF-FP-NOT: .cfi_endproc
; DWARF-FP:    .size test4,

; DWARF-FP-ELIM-LABEL: test4:
; DWARF-FP-ELIM-NOT: .cfi_startproc
; DWARF-FP-ELIM:     mov pc, lr
; DWARF-FP-ELIM-NOT: .cfi_endproc
; DWARF-FP-ELIM:     .size test4,

; DWARF-V7-FP-LABEL: test4:
; DWARF-V7-FP-NOT: .cfi_startproc
; DWARF-V7-FP:    bx lr
; DWARF-V7-FP-NOT: .cfi_endproc
; DWARF-V7-FP:    .size test4,

; DWARF-V7-FP-ELIM-LABEL: test4:
; DWARF-V7-FP-ELIM-NOT: .cfi_startproc
; DWARF-V7-FP-ELIM:     bx lr
; DWARF-V7-FP-ELIM-NOT: .cfi_endproc
; DWARF-V7-FP-ELIM:     .size test4,

; DWARF-WIN-FP-ELIM-LABEL: test4:
; DWARF-WIN-FP-ELIM-NOT: .cfi_startproc
; DWARF-WIN-FP-ELIM:     bx lr
; DWARF-WIN-FP-ELIM-NOT: .cfi_endproc

; SEH-WIN-FP-ELIM-LABEL: test4:
; SEH-WIN-FP-ELIM-NOT: .seh_proc test4
; SEH-WIN-FP-ELIM:     bx lr
; SEH-WIN-FP-ELIM-NOT: .seh_endproc
