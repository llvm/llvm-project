; RUN: llc %s -mtriple=mipsel -mcpu=mips32r2 -mattr=micromips -filetype=asm \
; RUN:   -relocation-model=static -o - | FileCheck %s

define i32 @sum(i32 %a, i32 %b) nounwind uwtable {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %1 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32, i32* %y, align 4
  %1 = load i32, i32* %z, align 4
  %call = call i32 @sum(i32 %0, i32 %1)
  store i32 %call, i32* %x, align 4
  %2 = load i32, i32* %x, align 4
  ret i32 %2
}

; CHECK:    .text

; CHECK:    .globl  sum
; CHECK:    .type sum,@function
; CHECK:    .set  micromips
; CHECK:    .ent  sum
; CHECK-LABEL: sum:
; CHECK:    .end  sum

; CHECK:    .globl  main
; CHECK:    .type main,@function
; CHECK:    .set  micromips
; CHECK:    .ent  main
; CHECK-LABEL: main:

; CHECK:    jal sum

; CHECK:    .end main
