; RUN: llc -mtriple=thumbv4t-none--eabi < %s | FileCheck %s --check-prefix=CHECK-V4T
; RUN: llc -mtriple=thumbv5t-none--eabi < %s | FileCheck %s --check-prefix=CHECK-V5T

; CHECK-V4T-LABEL: clobberframe
; CHECK-V5T-LABEL: clobberframe
define <4 x i32> @clobberframe(<6 x i32>* %p) #0 {
entry:
; Prologue
; --------
; CHECK-V4T:    push {[[SAVED:(r[4567](, )?)+]], lr}
; CHECK-V4T:    sub sp,
; CHECK-V5T:    push {[[SAVED:(r[4567](, )?)+]], lr}

  %b = alloca <6 x i32>, align 16
  %a = alloca <4 x i32>, align 16
  %stuff = load <6 x i32>, <6 x i32>* %p, align 16
  store <6 x i32> %stuff, <6 x i32>* %b, align 16
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <4 x i32>* %a, align 16
  %0 = load <4 x i32>, <4 x i32>* %a, align 16
  ret <4 x i32> %0

; Epilogue
; --------
; CHECK-V4T:         add sp,
; CHECK-V4T-NEXT:    pop {[[SAVED]]}
; The ISA for v4 does not support pop pc, so make sure we do not emit
; one even when we do not need to update SP.
; CHECK-V4T-NOT:     pop {pc}
; We may only use lo register to pop, but in that case, all the scratch
; ones are used.
; r12 is the only register we are allowed to clobber for AAPCS.
; Use it to save a lo register.
; CHECK-V4T-NEXT:    mov [[TEMP_REG:r12]], [[POP_REG:r[0-7]]]
; Pop the value of LR.
; CHECK-V4T-NEXT:    pop {[[POP_REG]]}
; Copy the value of LR in the right register.
; CHECK-V4T-NEXT:    mov lr, [[POP_REG]]
; Restore the value that was in the register we used to pop the value of LR.
; CHECK-V4T-NEXT:    mov [[POP_REG]], [[TEMP_REG]]
; Return.
; CHECK-V4T-NEXT:    bx lr
; CHECK-V5T:         pop {[[SAVED]], pc}
}

; CHECK-V4T-LABEL: clobbervariadicframe
; CHECK-V5T-LABEL: clobbervariadicframe
define <4 x i32> @clobbervariadicframe(i32 %i, ...) #0 {
entry:
; Prologue
; --------
; CHECK-V4T:    sub sp,
; CHECK-V4T:    push {[[SAVED:(r[4567](, )?)+]], lr}
; CHECK-V5T:    sub sp,
; CHECK-V5T:    push {[[SAVED:(r[4567](, )?)+]], lr}

  %b = alloca <4 x i32>, align 16
  %a = alloca <4 x i32>, align 16
  store <4 x i32> <i32 42, i32 42, i32 42, i32 42>, <4 x i32>* %b, align 16
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <4 x i32>* %a, align 16
  %0 = load <4 x i32>, <4 x i32>* %a, align 16
  call void @llvm.va_start(i8* null)
  ret <4 x i32> %0

; Epilogue
; --------
; CHECK-V4T:         pop {[[SAVED]]}
; CHECK-V4T-NEXT:    mov r12, [[POP_REG:r[0-7]]]
; CHECK-V4T-NEXT:    pop {[[POP_REG]]}
; CHECK-V4T-NEXT:    add sp,
; CHECK-V4T-NEXT:    mov lr, [[POP_REG]]
; CHECK-V4T-NEXT:    mov [[POP_REG]], r12
; CHECK-V4T:         bx  lr
; CHECK-V5T:         add sp,
; CHECK-V5T-NEXT:    pop {[[SAVED]]}
; CHECK-V5T-NEXT:    mov r12, [[POP_REG:r[0-7]]]
; CHECK-V5T-NEXT:    pop {[[POP_REG]]}
; CHECK-V5T-NEXT:    add sp,
; CHECK-V5T-NEXT:    mov lr, [[POP_REG]]
; CHECK-V5T-NEXT:    mov [[POP_REG]], r12
; CHECK-V5T-NEXT:    bx lr
}

; CHECK-V4T-LABEL: simpleframe
; CHECK-V5T-LABEL: simpleframe
define i32 @simpleframe(<6 x i32>* %p) #0 {
entry:
; Prologue
; --------
; CHECK-V4T:    push    {[[SAVED:(r[4567](, )?)+]], lr}
; CHECK-V5T:    push    {[[SAVED:(r[4567](, )?)+]], lr}

  %0 = load <6 x i32>, <6 x i32>* %p, align 16
  %1 = extractelement <6 x i32> %0, i32 0
  %2 = extractelement <6 x i32> %0, i32 1
  %3 = extractelement <6 x i32> %0, i32 2
  %4 = extractelement <6 x i32> %0, i32 3
  %5 = extractelement <6 x i32> %0, i32 4
  %6 = extractelement <6 x i32> %0, i32 5
  %add1 = add nsw i32 %1, %2
  %add2 = add nsw i32 %add1, %3
  %add3 = add nsw i32 %add2, %4
  %add4 = add nsw i32 %add3, %5
  %add5 = add nsw i32 %add4, %6
  ret i32 %add5

; Epilogue
; --------
; CHECK-V4T:    pop {[[SAVED]]}
; The ISA for v4 does not support pop pc, so make sure we do not emit
; one even when we do not need to update SP.
; CHECK-V4T-NOT:     pop {pc}
; Pop the value of LR into a scratch lo register other than r0 (it is
; used for the return value).
; CHECK-V4T-NEXT:    pop {[[POP_REG:r[1-3]]]}
; CHECK-V4T-NEXT:    bx [[POP_REG]]
; CHECK-V5T:    pop {[[SAVED]], pc}
}

; CHECK-V4T-LABEL: simplevariadicframe
; CHECK-V5T-LABEL: simplevariadicframe
define i32 @simplevariadicframe(i32 %i, ...) #0 {
entry:
; Prologue
; --------
; CHECK-V4T:    sub sp,
; CHECK-V4T:    push {[[SAVED:(r[4567](, )?)+]], lr}
; CHECK-V4T:    sub sp,
; CHECK-V5T:    sub sp,
; CHECK-V5T:    push {[[SAVED:(r[4567](, )?)+]], lr}
; CHECK-V5T:    sub sp,

  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  store i32 1, i32* %a, align 4
  store i32 2, i32* %b, align 4
  store i32 3, i32* %c, align 4
  store i32 4, i32* %d, align 4
  %0 = load i32, i32* %a, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %inc1 = add nsw i32 %1, 1
  store i32 %inc1, i32* %b, align 4
  %2 = load i32, i32* %c, align 4
  %inc2 = add nsw i32 %2, 1
  store i32 %inc2, i32* %c, align 4
  %3 = load i32, i32* %d, align 4
  %inc3 = add nsw i32 %3, 1
  store i32 %inc3, i32* %d, align 4
  %4 = load i32, i32* %a, align 4
  %5 = load i32, i32* %b, align 4
  %add = add nsw i32 %4, %5
  %6 = load i32, i32* %c, align 4
  %add4 = add nsw i32 %add, %6
  %7 = load i32, i32* %d, align 4
  %add5 = add nsw i32 %add4, %7
  %add6 = add nsw i32 %add5, %i
  call void @llvm.va_start(i8* null)
  ret i32 %add6

; Epilogue
; --------
; CHECK-V4T:         add sp,
; CHECK-V4T-NEXT:    pop {[[SAVED]]}
; Only r1 to r3 are available to pop LR.
; r0 is used for the return value.
; CHECK-V4T-NEXT:    pop {[[POP_REG:r[1-3]]]}
; CHECK-V4T-NEXT:    add sp,
; CHECK-V4T-NEXT:    bx [[POP_REG]]
; CHECK-V5T:         add sp,
; CHECK-V5T-NEXT:    pop {[[SAVED]]}
; Only r1 to r3 are available to pop LR.
; r0 is used for the return value.
; CHECK-V5T-NEXT:    pop {[[POP_REG:r[1-3]]]}
; CHECK-V5T-NEXT:    add sp,
; CHECK-V5T-NEXT:    bx [[POP_REG]]
}

; CHECK-V4T-LABEL: noframe
; CHECK-V5T-LABEL: noframe
define i32 @noframe() #0 {
entry:
; Prologue
; --------
; CHECK-V4T-NOT: push
; CHECK-V5T-NOT: push
    ret i32 0;
; Epilogue
; --------
; CHECK-V4T-NOT: pop
; CHECK-V5T-NOT: pop
; CHECK-V4T:    bx  lr
; CHECK-V5T:    bx  lr
}

; CHECK-V4T-LABEL: novariadicframe
; CHECK-V5T-LABEL: novariadicframe
define i32 @novariadicframe(i32 %i, ...) #0 {
entry:
; Prologue
; --------
; CHECK-V4T:    sub sp,
; CHECK-V4T:    push {[[SAVED:(r[4567](, )?)+]], lr}
; CHECK-V5T:    sub sp,
; CHECK-V5T:    push {[[SAVED:(r[4567](, )?)+]], lr}

  call void @llvm.va_start(i8* null)
  ret i32 %i;
; Epilogue
; --------
; CHECK-V4T:         pop {[[SAVED]]}
; Only r1 to r3 are available to pop LR.
; r0 is used for the return value.
; CHECK-V4T-NEXT:    pop {[[POP_REG:r[1-3]]]}
; CHECK-V4T-NEXT:    add sp,
; CHECK-V4T-NEXT:    bx [[POP_REG]]
; CHECK-V5T:         pop {[[SAVED]]}
; Only r1 to r3 are available to pop LR.
; r0 is used for the return value.
; CHECK-V5T-NEXT:    pop {[[POP_REG:r[1-3]]]}
; CHECK-V5T-NEXT:    add sp,
; CHECK-V5T-NEXT:    bx [[POP_REG]]
}

declare void @llvm.va_start(i8*) nounwind
