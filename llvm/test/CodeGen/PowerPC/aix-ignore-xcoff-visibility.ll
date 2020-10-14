; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -data-sections=false < %s | \
; RUN:   FileCheck --check-prefix=VISIBILITY-ASM %s
; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -data-sections=false -ignore-xcoff-visibility < %s | \
; RUN:   FileCheck --check-prefix=IGNOREVISIBILITY-ASM %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -data-sections=false < %s | \
; RUN:   FileCheck --check-prefix=VISIBILITY-ASM %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -data-sections=false -ignore-xcoff-visibility < %s | \
; RUN:   FileCheck --check-prefix=IGNOREVISIBILITY-ASM %s

@foo_p = global void ()* @zoo_extern_h, align 4
@b = protected global i32 0, align 4

define hidden void @foo_h(i32* %p) {
entry:
  %p.addr = alloca i32*, align 4
  store i32* %p, i32** %p.addr, align 4
  %0 = load i32*, i32** %p.addr, align 4
  %1 = load i32, i32* %0, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* %0, align 4
  ret void
}

declare hidden void @zoo_extern_h()

define protected void @bar() {
entry:
  call void @foo_h(i32* @b)
  %0 = load void ()*, void ()** @foo_p, align 4
  call void %0()
  ret void
}

; VISIBILITY-ASM: .globl  foo_h[DS],hidden
; VISIBILITY-ASM: .globl  .foo_h,hidden
; VISIBILITY-ASM: .globl  bar[DS],protected
; VISIBILITY-ASM: .globl  .bar,protected
; VISIBILITY-ASM: .globl  b,protected

; IGNOREVISIBILITY-ASM: .globl  foo_h[DS]
; IGNOREVISIBILITY-ASM: .globl  .foo_h
; IGNOREVISIBILITY-ASM: .globl  bar[DS]
; IGNOREVISIBILITY-ASM: .globl  .bar
; IGNOREVISIBILITY-ASM: .globl  b
