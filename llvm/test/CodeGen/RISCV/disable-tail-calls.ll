; Check that command line option "-disable-tail-calls" overrides function
; attribute "disable-tail-calls".

; RUN: llc < %s -mtriple=riscv32-unknown-elf \
; RUN: | FileCheck %s --check-prefixes=CALLER1,NOTAIL
; RUN: llc < %s -mtriple=riscv32-unknown-elf -disable-tail-calls \
; RUN: | FileCheck %s --check-prefixes=CALLER1,NOTAIL
; RUN: llc < %s -mtriple=riscv32-unknown-elf -disable-tail-calls=false \
; RUN: | FileCheck %s --check-prefixes=CALLER1,TAIL

; RUN: llc < %s -mtriple=riscv32-unknown-elf \
; RUN: | FileCheck %s --check-prefixes=CALLER2,TAIL
; RUN: llc < %s -mtriple=riscv32-unknown-elf -disable-tail-calls \
; RUN: | FileCheck %s --check-prefixes=CALLER2,NOTAIL
; RUN: llc < %s -mtriple=riscv32-unknown-elf -disable-tail-calls=false \
; RUN: | FileCheck %s --check-prefixes=CALLER2,TAIL

; RUN: llc < %s -mtriple=riscv32-unknown-elf \
; RUN: | FileCheck %s --check-prefixes=CALLER3,TAIL
; RUN: llc < %s -mtriple=riscv32-unknown-elf -disable-tail-calls \
; RUN: | FileCheck %s --check-prefixes=CALLER3,NOTAIL
; RUN: llc < %s -mtriple=riscv32-unknown-elf -disable-tail-calls=false \
; RUN: | FileCheck %s --check-prefixes=CALLER3,TAIL

; CALLER1-LABEL: {{\_?}}caller1
; CALLER2-LABEL: {{\_?}}caller2
; CALLER3-LABEL: {{\_?}}caller3
; NOTAIL-NOT: tail callee
; NOTAIL: call callee
; TAIL: tail callee
; TAIL-NOT: call callee

; Function with attribute #0 = { "disable-tail-calls"="true" }
define i32 @caller1(i32 %a) #0 {
entry:
  %call = tail call i32 @callee(i32 %a)
  ret i32 %call
}

; Function with attribute #1 = { "disable-tail-calls"="false" }
define i32 @caller2(i32 %a) #0 {
entry:
  %call = tail call i32 @callee(i32 %a)
  ret i32 %call
}

define i32 @caller3(i32 %a) {
entry:
  %call = tail call i32 @callee(i32 %a)
  ret i32 %call
}

declare i32 @callee(i32)

attributes #0 = { "disable-tail-calls"="true" }
attributes #1 = { "disable-tail-calls"="false" }
