; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec -data-sections=false < %s | \
; RUN:   FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec -data-sections=false < %s |\
; RUN:   FileCheck %s

; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -filetype=obj -o %t.o < %s 2>&1 | \
; RUN:   FileCheck --check-prefix=XCOFF32 %s
; XCOFF32: LLVM ERROR: Emitting non-zero visibilities is not supported yet.

; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -filetype=obj -o %t.o 2>&1 < %s 2>&1 | \
; RUN:   FileCheck --check-prefix=XCOFF64 %s
; FIXME: This should check for the visibility error, but we actually fail before
; that due to unimplemented relocations.
; XCOFF64: LLVM ERROR: Unimplemented fixup kind.

@b =  global i32 0, align 4
@b_h = hidden global i32 0, align 4
@b_e = dllexport global i32 0, align 4

define void @foo() {
entry:
  ret void
}

define hidden void @foo_h(i32* %ip) {
entry:
  ret void
}

define dllexport void @foo_e(i32* %ip) {
entry:
  ret void
}

define protected void @foo_protected(i32* %ip) {
entry:
  ret void
}

define weak hidden void @foo_weak_h() {
entry:
  ret void
}

define weak dllexport void @foo_weak_e() {
entry:
  ret void
}

@foo_p = global void ()* @zoo_weak_extern_h, align 4
declare extern_weak hidden void @zoo_weak_extern_h()
declare extern_weak dllexport void @zoo_weak_extern_e()

define i32 @main() {
entry:
  %call1= call i32 @bar_h(i32* @b_h)
  call void @foo_weak_h()
  %0 = load void ()*, void ()** @foo_p, align 4
  call void %0()
  ret i32 0
}

declare hidden i32 @bar_h(i32*)
declare dllexport i32 @bar_e(i32*)

; CHECK:        .globl  foo[DS]{{[[:space:]]*([#].*)?$}}
; CHECK:        .globl  .foo{{[[:space:]]*([#].*)?$}}
; CHECK:        .globl  foo_h[DS],hidden
; CHECK:        .globl  .foo_h,hidden
; CHECK:        .globl  foo_e[DS],exported
; CHECK:        .globl  .foo_e,exported
; CHECK:        .globl  foo_protected[DS],protected
; CHECK:        .globl  .foo_protected,protected
; CHECK:        .weak   foo_weak_h[DS],hidden
; CHECK:        .weak   .foo_weak_h,hidden
; CHECK:        .weak   foo_weak_e[DS],exported
; CHECK:        .weak   .foo_weak_e,exported

; CHECK:        .globl  b{{[[:space:]]*([#].*)?$}}
; CHECK:        .globl  b_h,hidden
; CHECK:        .globl  b_e,exported

; CHECK:        .weak   .zoo_weak_extern_h[PR],hidden
; CHECK:        .weak   zoo_weak_extern_h[DS],hidden
; CHECK:        .weak   .zoo_weak_extern_e[PR],exported
; CHECK:        .weak   zoo_weak_extern_e[DS],exported
; CHECK:        .extern .bar_h[PR],hidden
; CHECK:        .extern bar_h[DS],hidden
; CHECK:        .extern .bar_e[PR],exported
; CHECK:        .extern bar_e[DS],exported
