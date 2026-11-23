; NOTE: Do not autogenerate
; RUN: llc -mtriple=aarch64-linux-gnu -O2 -verify-machineinstrs -debug %s -o - 2>&1 | FileCheck %s --check-prefix=GISEL
; REQUIRES: asserts

; Check both an optnone function and a non-optnone function to ensure that only
; the optnone functions is routed through GlobalISel

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

; There are two functions being checked here, one which is optnone and the other which isn't.
; As we are running at O2, we default to using SDAG for selection. The optnone function however,
; will make use of GlobalISel. The top check, checks that the GlobalISel pipeline is run on the 
; optnone function, and the second line is checking that SDAG was run on the normal function.

; GISEL: Skipping pass 'IRTranslator' on function optnone_fn
; GISEL: Creating new node:

attributes #0 = { noinline nounwind optnone uwtable }
