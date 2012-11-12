; Ensure that the MBlaze interrupt_handler calling convention (cc73) is handled
; correctly correctly by the MBlaze backend.
;
; RUN: llc < %s -march=mblaze | FileCheck %s

@.str = private constant [28 x i8] c"The interrupt has gone off\0A\00"
@_interrupt_handler = alias void ()* @myintr

define cc73 void @myintr() nounwind noinline {
  ; CHECK:        myintr:
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
  ; CHECK:        mfs   r11, rmsr
  ; CHECK:        swi   r11, r1
  entry:
    %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([28 x i8]* @.str, i32 0, i32 0))
      ret void

  ; CHECK:        lwi   r11, r1
  ; CHECK:        mts   rmsr, r11
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
  ; CHECK:        rtid  r14, 0
}

  ; CHECK:    .globl  _interrupt_handler
  ; CHECK:    _interrupt_handler = myintr
declare i32 @printf(i8*, ...)
