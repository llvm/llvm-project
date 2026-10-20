; RUN: llc -O0 -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

; Checks that fake uses of the FP stack do not cause a crash.
;
; /*******************************************************************/
; extern long double foo(long double, long double, long double);
;
; long double actual(long double p1, long double p2, long double p3) {
;   return fmal(p1, p2, p3);
; }
; /*******************************************************************/

define x86_fp80 @actual(x86_fp80 %p1, x86_fp80 %p2, x86_fp80 %p3) optdebug {
;
; CHECK: actual
;
entry:
  %p1.addr = alloca x86_fp80, align 16
  %p2.addr = alloca x86_fp80, align 16
  %p3.addr = alloca x86_fp80, align 16
  store x86_fp80 %p1, ptr %p1.addr, align 16
  store x86_fp80 %p2, ptr %p2.addr, align 16
  store x86_fp80 %p3, ptr %p3.addr, align 16
  %0 = load x86_fp80, ptr %p1.addr, align 16
  %1 = load x86_fp80, ptr %p2.addr, align 16
  %2 = load x86_fp80, ptr %p3.addr, align 16
;
; CHECK: callq{{.*}}foo
;
  %3 = call x86_fp80 @foo(x86_fp80 %0, x86_fp80 %1, x86_fp80 %2)
  %4 = load x86_fp80, ptr %p1.addr, align 16
  call void (...) @llvm.fake.use(x86_fp80 %4)
  %5 = load x86_fp80, ptr %p2.addr, align 16
  call void (...) @llvm.fake.use(x86_fp80 %5)
  %6 = load x86_fp80, ptr %p3.addr, align 16
  call void (...) @llvm.fake.use(x86_fp80 %6)
;
; CHECK: ret
;
  ret x86_fp80 %3
}

declare x86_fp80 @foo(x86_fp80, x86_fp80, x86_fp80)
