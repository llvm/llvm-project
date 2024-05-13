; RUN: llc < %s -mtriple=avr -mcpu=avr6 | FileCheck %s --check-prefixes=CHECK,AVR6
; RUN: llc < %s -mtriple=avr -mcpu=avr2 | FileCheck %s --check-prefixes=CHECK,AVR2

; TODO: test returning byval structs

declare i8 @foo8_1(i8)
declare i8 @foo8_2(i8, i8, i8)
declare i8 @foo8_3(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)

declare i16 @foo16_1(i16, i16)
declare i16 @foo16_2(i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16)

declare i32 @foo32_1(i32, i32)
declare i32 @foo32_2(i32, i32, i32, i32, i32)

declare i64 @foo64_1(i64)
declare i64 @foo64_2(i64, i64, i64)

define i8 @calli8_reg() {
; CHECK-LABEL: calli8_reg:
; CHECK: ldi r24, 12
; AVR6:  call foo8_1
; AVR2:  rcall foo8_1
; CHECK: ldi r24, 12
; CHECK: ldi r22, 13
; CHECK: ldi r20, 14
; AVR6:  call foo8_2
; AVR2:  rcall foo8_2
    %result1 = call i8 @foo8_1(i8 12)
    %result2 = call i8 @foo8_2(i8 12, i8 13, i8 14)
    ret i8 %result2
}

define i8 @calli8_stack() {
; CHECK-LABEL: calli8_stack:
; CHECK: ldi [[REG1:r[0-9]+]], 10
; CHECK: ldi [[REG2:r[0-9]+]], 11
; CHECK: std Z+2, [[REG2]]
; CHECK: std Z+1, [[REG1]]
; AVR6:  call foo8_3
; AVR2:  rcall foo8_3
    %result1 = call i8 @foo8_3(i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11)
    ret i8 %result1
}

define i16 @calli16_reg() {
; CHECK-LABEL: calli16_reg:
; CHECK: ldi r24, 1
; CHECK: ldi r25, 2
; CHECK: ldi r22, 2
; CHECK: ldi r23, 2
; AVR6:  call foo16_1
; AVR2:  rcall foo16_1
    %result1 = call i16 @foo16_1(i16 513, i16 514)
    ret i16 %result1
}

define i16 @calli16_stack() {
; CHECK-LABEL: calli16_stack:
; CHECK: ldi [[REG1:r[0-9]+]], 10
; CHECK: ldi [[REG2:r[0-9]+]], 2
; CHECK: std Z+4, [[REG2]]
; CHECK: std Z+3, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 9
; CHECK: ldi [[REG2:r[0-9]+]], 2
; CHECK: std Z+2, [[REG2]]
; CHECK: std Z+1, [[REG1]]
; AVR6:  call foo16_2
; AVR2:  rcall foo16_2
    %result1 = call i16 @foo16_2(i16 512, i16 513, i16 514, i16 515, i16 516, i16 517, i16 518, i16 519, i16 520, i16 521, i16 522)
    ret i16 %result1
}

define i32 @calli32_reg() {
; CHECK-LABEL: calli32_reg:
; CHECK: ldi r22, 64
; CHECK: ldi r23, 66
; CHECK: ldi r24, 15
; CHECK: ldi r25, 2
; CHECK: ldi r18, 128
; CHECK: ldi r19, 132
; CHECK: ldi r20, 30
; CHECK: ldi r21, 2
; AVR6:  call foo32_1
; AVR2:  rcall foo32_1
    %result1 = call i32 @foo32_1(i32 34554432, i32 35554432)
    ret i32 %result1
}

define i32 @calli32_stack() {
; CHECK-LABEL: calli32_stack:
; CHECK: ldi [[REG1:r[0-9]+]], 15
; CHECK: ldi [[REG2:r[0-9]+]], 2
; CHECK: std Z+4, [[REG2]]
; CHECK: std Z+3, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 64
; CHECK: ldi [[REG2:r[0-9]+]], 66
; CHECK: std Z+2, [[REG2]]
; CHECK: std Z+1, [[REG1]]
; AVR6:  call foo32_2
; AVR2:  rcall foo32_2
    %result1 = call i32 @foo32_2(i32 1, i32 2, i32 3, i32 4, i32 34554432)
    ret i32 %result1
}

define i64 @calli64_reg() {
; CHECK-LABEL: calli64_reg:
; CHECK: ldi r18, 255
; CHECK: ldi r19, 255
; CHECK: ldi r20, 155
; CHECK: ldi r21, 88
; CHECK: ldi r22, 76
; CHECK: ldi r23, 73
; CHECK: ldi r24, 31
; CHECK: ldi r25, 242
; AVR6:  call foo64_1
; AVR2:  rcall foo64_1
    %result1 = call i64 @foo64_1(i64 17446744073709551615)
    ret i64 %result1
}

define i64 @calli64_stack() {
; CHECK-LABEL: calli64_stack:

; CHECK: ldi [[REG1:r[0-9]+]], 31
; CHECK: ldi [[REG2:r[0-9]+]], 242
; CHECK: std Z+8, [[REG2]]
; CHECK: std Z+7, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 76
; CHECK: ldi [[REG2:r[0-9]+]], 73
; CHECK: std Z+6, [[REG2]]
; CHECK: std Z+5, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 155
; CHECK: ldi [[REG2:r[0-9]+]], 88
; CHECK: std Z+4, [[REG2]]
; CHECK: std Z+3, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 255
; CHECK: ldi [[REG2:r[0-9]+]], 255
; CHECK: std Z+2, [[REG2]]
; CHECK: std Z+1, [[REG1]]
; AVR6:  call foo64_2
; AVR2:  rcall foo64_2
    %result1 = call i64 @foo64_2(i64 1, i64 2, i64 17446744073709551615)
    ret i64 %result1
}

; Test passing arguments through the stack when the call frame is allocated
; in the prologue.
declare void @foo64_3(i64, i64, i64, i8, ptr)

define void @testcallprologue() {
; CHECK-LABEL: testcallprologue:
; CHECK: push r28
; CHECK: push r29
; CHECK: sbiw r28, 27
; CHECK: ldi [[REG1:r[0-9]+]], 88
; CHECK: std Y+9, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 11
; CHECK: ldi [[REG2:r[0-9]+]], 10
; CHECK: std Y+8, [[REG2]]
; CHECK: std Y+7, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 13
; CHECK: ldi [[REG2:r[0-9]+]], 12
; CHECK: std Y+6, [[REG2]]
; CHECK: std Y+5, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 15
; CHECK: ldi [[REG2:r[0-9]+]], 14
; CHECK: std Y+4, [[REG2]]
; CHECK: std Y+3, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 8
; CHECK: ldi [[REG2:r[0-9]+]], 9
; CHECK: std Y+2, [[REG2]]
; CHECK: std Y+1, [[REG1]]
; CHECK: pop r29
; CHECK: pop r28
  %p = alloca [8 x i16]
  %arraydecay = getelementptr inbounds [8 x i16], ptr %p, i16 0, i16 0
  call void @foo64_3(i64 723685415333071112, i64 723685415333071112, i64 723685415333071112, i8 88, ptr %arraydecay)
  ret void
}

define i32 @icall(ptr addrspace(1) %foo) {
; CHECK-LABEL: icall:
; AVR6:  movw r30, r24
; AVR2:  mov r30, r24
; AVR2:  mov r31, r25
; CHECK: ldi r22, 147
; CHECK: ldi r23, 248
; CHECK: ldi r24, 214
; CHECK: ldi r25, 198
; CHECK: icall
; CHECK: subi r22, 251
; CHECK: sbci r23, 255
; CHECK: sbci r24, 255
; CHECK: sbci r25, 255
  %1 = call i32 %foo(i32 3335977107)
  %2 = add nsw i32 %1, 5
  ret i32 %2
}

; Calling external functions (like __divsf3) require extra processing for
; arguments and return values in the LowerCall function.
declare i32 @foofloat(float)

define i32 @externcall(float %a, float %b) {
; CHECK-LABEL: externcall:
; AVR6:  movw [[REG0:(r[0-9]+|[XYZ])]], r24
; AVR6:  movw [[REG1:(r[0-9]+|[XYZ])]], r22
; AVR6:  movw r22, r18
; AVR6:  movw r24, r20
; AVR6:  movw r18, [[REG1]]
; AVR6:  movw r20, [[REG0]]
; AVR6:  call __divsf3
; AVR6:  call foofloat
; AVR2:  mov [[REG0:(r[0-9]+)]], r24
; AVR2:  mov [[REG1:(r[0-9]+)]], r25
; AVR2:  mov [[REG2:(r[0-9]+)]], r22
; AVR2:  mov [[REG3:(r[0-9]+)]], r23
; AVR2:  mov r22, r18
; AVR2:  mov r23, r19
; AVR2:  mov r24, r20
; AVR2:  mov r25, r21
; AVR2:  mov r18, [[REG2]]
; AVR2:  mov r19, [[REG3]]
; AVR2:  mov r20, [[REG0]]
; AVR2:  mov r21, [[REG1]]
; AVR2:  rcall __divsf3
; AVR2:  rcall foofloat
; CHECK: subi r22, 251
; CHECK: sbci r23, 255
; CHECK: sbci r24, 255
; CHECK: sbci r25, 255
  %1 = fdiv float %b, %a
  %2 = call i32 @foofloat(float %1)
  %3 = add nsw i32 %2, 5
  ret i32 %3
}
