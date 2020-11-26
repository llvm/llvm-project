; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs -mattr=+outline-atomics < %s | FileCheck %s --check-prefix=OUTLINE-ATOMICS

define i32 @foo(i32* %var, i1 %cond) {
; OUTLINE-ATOMICS: bl __aarch64_ldadd4_relax
; CHECK-LABEL: foo:
  br i1 %cond, label %atomic_ver, label %simple_ver
simple_ver:
  %oldval = load i32, i32* %var
  %newval = add nsw i32 %oldval, -1
  store i32 %newval, i32* %var
  br label %somewhere
atomic_ver:
  fence seq_cst
  %val = atomicrmw add i32* %var, i32 -1 monotonic
  fence seq_cst
  br label %somewhere
; CHECK: dmb
; CHECK: ldxr
; CHECK: dmb
  ; The key point here is that the second dmb isn't immediately followed by the
  ; simple_ver basic block, which LLVM attempted to do when DMB had been marked
  ; with isBarrier. For now, look for something that looks like "somewhere".
; CHECK-NEXT: mov
somewhere:
  %combined = phi i32 [ %val, %atomic_ver ], [ %newval, %simple_ver]
  ret i32 %combined
}
