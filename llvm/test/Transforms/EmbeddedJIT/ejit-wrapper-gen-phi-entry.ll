; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s
; RUN: opt -passes=ejit-aot-module,lower-expect -S %s -o /dev/null
; RUN: opt -passes='default<O2>' -S %s -o /dev/null

; Regression test for a crash in EJitWrapperGen: when the ejit_entry function's
; entry block is a PHI incoming predecessor, the wrapper rewrite erased the
; entry block without updating PHI incoming blocks, leaving dangling pointers
; that crashed lower-expect (and thus any -O1/-O2/-O3 build of such a function).
;
; The trigger is short-circuit && / || wrapped in __builtin_expect, e.g.:
;     #define MC_LIKELY(x) (__builtin_expect((!!(x)), 1))
;     if (MC_LIKELY(a > 0 && b > 0)) { ... }
; The short-circuit && emits a PHI in the merge block whose incoming block is
; the entry block; __builtin_expect feeds that PHI to llvm.expect, so
; lower-expect's handlePhiDef dereferences the dangling pointer.

; --- single-dim ejit_entry with a short-circuit && (PHI in merge block) ---
; After wrapper-gen the merge PHI must have jit_fallback as its incoming block,
; NOT the erased entry / jit_entry.
define i32 @entry_short_circuit_and(i32 %a, i32 %b) !ejit.metadata !0 {
entry:
  %cmp.a = icmp sgt i32 %a, 0
  br i1 %cmp.a, label %rhs, label %merge

rhs:
  %cmp.b = icmp sgt i32 %b, 0
  br label %merge

merge:
  %and = phi i1 [ false, %entry ], [ %cmp.b, %rhs ]
  %norm = xor i1 %and, true
  %norm2 = xor i1 %norm, true
  %z = zext i1 %norm2 to i32
  %sext = sext i32 %z to i64
  %exp = call i64 @llvm.expect.i64(i64 %sext, i64 1)
  %tobool = icmp ne i64 %exp, 0
  br i1 %tobool, label %then, label %else

then:
  ret i32 1

else:
  ret i32 0
}

; --- short-circuit || : same pattern, second incoming value is true ---
define i32 @entry_short_circuit_or(i32 %a, i32 %b) !ejit.metadata !0 {
entry:
  %cmp.a = icmp sgt i32 %a, 0
  br i1 %cmp.a, label %merge, label %rhs

rhs:
  %cmp.b = icmp sgt i32 %b, 0
  br label %merge

merge:
  %or = phi i1 [ true, %entry ], [ %cmp.b, %rhs ]
  %norm = xor i1 %or, true
  %norm2 = xor i1 %norm, true
  %z = zext i1 %norm2 to i32
  %sext = sext i32 %z to i64
  %exp = call i64 @llvm.expect.i64(i64 %sext, i64 1)
  %tobool = icmp ne i64 %exp, 0
  br i1 %tobool, label %then, label %else

then:
  ret i32 1

else:
  ret i32 0
}

declare i64 @llvm.expect.i64(i64, i64)

; ejit_entry metadata (no period_arr_ind — static-only dependency).
!0 = distinct !{!{!"ejit_entry"}}

; CHECK-LABEL: define i32 @entry_short_circuit_and
; CHECK: jit_entry:
; The merge PHI's incoming block referencing the original entry must now point
; at jit_fallback (the block the entry body was spliced into), not jit_entry
; and not the erased entry block.
; CHECK: %and = phi i1 [ false, %jit_fallback ], [ %cmp.b, %rhs ]
; CHECK: jit_fallback:

; CHECK-LABEL: define i32 @entry_short_circuit_or
; CHECK: %or = phi i1 [ true, %jit_fallback ], [ %cmp.b, %rhs ]
