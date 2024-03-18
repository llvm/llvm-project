; RUN: llc -mtriple s390x-ibm-zos < %s | FileCheck %s

; C test case:
; typedef void (*voidfunc)();
; void func(){}
; voidfunc ptr = &func;

; CHECK: ptr:
; CHECK:        .quad   func

@ptr = hidden global void (...)* bitcast (void ()* @func to void (...)*), align 8
  
define hidden void @func() {
entry:
  ret void
}

; C test case for global alias
; void __f() {}
; void f() __attribute__ ((alias ("__f")));
; void (*fp)() = &f;

; CHECK:        .quad   f

; CHECK:        .globl  f
; CHECK:        .hidden f
; CHECK: .set f, V(__f)

@fp = hidden global void (...)* @f, align 8
@f = hidden alias void (...), bitcast (void ()* @__f to void (...)*)

define hidden void @__f() #0 {
entry:
  ret void
}
