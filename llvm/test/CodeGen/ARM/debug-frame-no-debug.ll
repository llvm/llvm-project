; ARM EHABI integrated test

; This test case checks that the ARM DWARF stack frame directives
; are not generated if compiling with no debug information.
  
; RUN: llc -mtriple arm-unknown-linux-gnueabi \
; RUN:     -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-FP-ELIM

; RUN: llc -mtriple thumb-unknown-linux-gnueabi \
; RUN:     -frame-pointer=all -filetype=asm -o - %s \
; RUN:   | FileCheck %s --check-prefix=CHECK-THUMB-FP

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

; CHECK-FP-ELIM-LABEL: _Z4testiiiiiddddd:
; CHECK-FP-ELIM-NOT:   .cfi_startproc
; CHECK-FP-ELIM:   push  {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-FP-ELIM-NOT:   .cfi_def_cfa_offset 36

; CHECK-THUMB-FP-LABEL: _Z4testiiiiiddddd:
; CHECK-THUMB-FP-NOT:   .cfi_startproc
; CHECK-THUMB-FP:   push   {r4, r5, r6, r7, lr}
; CHECK-THUMB-FP-NOT:   .cfi_def_cfa_offset 20

