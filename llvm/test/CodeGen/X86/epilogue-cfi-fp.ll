; RUN: llc -O0 %s -o - | FileCheck %s
; RUN: llc -O0 %s -o - | FileCheck %s
; RUN: llc < %s -stop-after=prologepilog | FileCheck %s --check-prefix=PEI

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i686-pc-linux"

; Function Attrs: noinline nounwind
define i32 @foo(i32 %i, i32 %j, i32 %k, i32 %l, i32 %m) #0 {

; CHECK-LABEL:   foo:
; CHECK:         popl %ebp
; CHECK-NEXT:    .cfi_def_cfa %esp, 4
; CHECK-NEXT:    retl

; PEI-LABEL: name: foo
; PEI:         $ebp = frame-destroy POP32r implicit-def $esp, implicit $esp
; PEI-NEXT:    frame-destroy CFI_INSTRUCTION def_cfa $esp, 4
; PEI-NEXT:    RET 0, killed $eax

entry:
  %i.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %l.addr = alloca i32, align 4
  %m.addr = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  store i32 %j, ptr %j.addr, align 4
  store i32 %k, ptr %k.addr, align 4
  store i32 %l, ptr %l.addr, align 4
  store i32 %m, ptr %m.addr, align 4
  ret i32 0
}

attributes #0 = { "frame-pointer"="all" }

