; Ensure that the MBlaze save_volatiles calling convention (cc74) is handled
; correctly correctly by the MBlaze backend.
;
; RUN: llc < %s -march=mblaze | FileCheck %s

@.str = private constant [28 x i8] c"The interrupt has gone off\0A\00"

define cc74 void @mysvol() nounwind noinline {
  ; CHECK:        mysvol:
  ; CHECK:        swi   r3, r1
  ; CHECK:        swi   r4, r1
  ; CHECK:        swi   r5, r1
  ; CHECK:        swi   r6, r1
  ; CHECK:        swi   r7, r1
  ; CHECK:        swi   r8, r1
  ; CHECK:        swi   r9, r1
  ; CHECK:        swi   r10, r1
  ; CHECK:        swi   r11, r1
  ; CHECK:        swi   r12, r1
  ; CHECK:        swi   r17, r1
  ; CHECK:        swi   r18, r1
  ; CHECK-NOT:    mfs   r11, rmsr
  entry:
    %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([28 x i8]* @.str, i32 0, i32 0))
      ret void

  ; CHECK-NOT:    mts   rmsr, r11
  ; CHECK:        lwi   r18, r1
  ; CHECK:        lwi   r17, r1
  ; CHECK:        lwi   r12, r1
  ; CHECK:        lwi   r11, r1
  ; CHECK:        lwi   r10, r1
  ; CHECK:        lwi   r9, r1
  ; CHECK:        lwi   r8, r1
  ; CHECK:        lwi   r7, r1
  ; CHECK:        lwi   r6, r1
  ; CHECK:        lwi   r5, r1
  ; CHECK:        lwi   r4, r1
  ; CHECK:        lwi   r3, r1
  ; CHECK:        rtsd  r15, 8
}

define cc74 void @mysvol2() nounwind noinline {
  ; CHECK:        mysvol2:
  ; CHECK-NOT:    swi   r3, r1
  ; CHECK-NOT:    swi   r4, r1
  ; CHECK-NOT:    swi   r5, r1
  ; CHECK-NOT:    swi   r6, r1
  ; CHECK-NOT:    swi   r7, r1
  ; CHECK-NOT:    swi   r8, r1
  ; CHECK-NOT:    swi   r9, r1
  ; CHECK-NOT:    swi   r10, r1
  ; CHECK-NOT:    swi   r11, r1
  ; CHECK-NOT:    swi   r12, r1
  ; CHECK:        swi   r17, r1
  ; CHECK:        swi   r18, r1
  ; CHECK-NOT:    mfs   r11, rmsr
entry:

  ; CHECK-NOT:    mts   rmsr, r11
  ; CHECK:        lwi   r18, r1
  ; CHECK:        lwi   r17, r1
  ; CHECK-NOT:    lwi   r12, r1
  ; CHECK-NOT:    lwi   r11, r1
  ; CHECK-NOT:    lwi   r10, r1
  ; CHECK-NOT:    lwi   r9, r1
  ; CHECK-NOT:    lwi   r8, r1
  ; CHECK-NOT:    lwi   r7, r1
  ; CHECK-NOT:    lwi   r6, r1
  ; CHECK-NOT:    lwi   r5, r1
  ; CHECK-NOT:    lwi   r4, r1
  ; CHECK-NOT:    lwi   r3, r1
  ; CHECK:        rtsd  r15, 8
  ret void
}

  ; CHECK-NOT:    .globl  _interrupt_handler
  ; CHECK-NOT:    _interrupt_handler = mysvol
  ; CHECK-NOT:    _interrupt_handler = mysvol2
declare i32 @printf(i8*, ...)
