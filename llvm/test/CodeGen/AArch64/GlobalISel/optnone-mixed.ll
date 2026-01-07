; NOTE: Do not autogenerate
; RUN: llc -mtriple=aarch64-linux-gnu -O2 -verify-machineinstrs -debug %s -o - 2>&1 | FileCheck %s --check-prefix=GISEL
; REQUIRES: asserts

; Check both an optnone function and a non-optnone function to ensure that only
; the optnone functions is routed through GlobalISel

; Function Attrs: noinline nounwind optnone uwtable
define i32 @optnone_fn(i32 %a) #0 {
entry:
  %add = add nsw i32 %a, 1
  ret i32 %add
}

define i32 @normal_fn(i32 %a) {
entry:
  %mul = mul nsw i32 %a, 2
  ret i32 %mul
}

; GISEL: Skipping pass 'AArch64PostLegalizerCombiner' on function optnone_fn
; GISEL: Creating new node:

attributes #0 = { noinline nounwind optnone uwtable }
