; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s
;
; XFAIL: *
;
; KNOWN BUG (recorded now, fix tracked separately).
;
; The jit_dispatch indirect call drops ALL parameter and return ABI attributes
; of the ejit_entry function. ejit_compile_or_get returns a function compiled
; from F's own bitcode (so its ABI honours byval/sret/signext/zeroext/...), but
; the dispatch call site carries none of them. On the target ABI (AArch64,
; Cortex-A) byval/sret and sub-word signext/zeroext are all part of the calling
; convention, so the mismatched call corrupts arguments / the return slot.
;
; The fix should copy F->getAttributes() onto the dispatch CallInst. Until then
; the CHECKs below (correct behavior) do not match -> XFAIL.

%S = type { i32, i32 }
%R = type { i64, i64 }

; --- byval struct argument must stay byval at the dispatch call ---
; CHECK-LABEL: define void @byval_entry(ptr byval(%S) %p, i8 %cell)
; CHECK: jit_dispatch:
; CHECK: call void {{.*}}(ptr byval(%S) %p, i8 %cell)
define void @byval_entry(ptr byval(%S) %p, i8 %cell) !ejit.metadata !1 {
entry:
  ret void
}

; --- sret pointer must stay sret at the dispatch call ---
; CHECK-LABEL: define void @sret_entry(ptr sret(%R) %out, i8 %cell)
; CHECK: jit_dispatch:
; CHECK: call void {{.*}}(ptr sret(%R) %out, i8 %cell)
define void @sret_entry(ptr sret(%R) %out, i8 %cell) !ejit.metadata !1 {
entry:
  ret void
}

; --- signext on the return value and on a sub-word argument must be preserved ---
; CHECK-LABEL: define signext i16 @signext_entry(i16 signext %a)
; CHECK: jit_dispatch:
; CHECK: call signext i16 {{.*}}(i16 signext %a)
define signext i16 @signext_entry(i16 signext %a) !ejit.metadata !0 {
entry:
  ret i16 %a
}

!0 = distinct !{!{!"ejit_entry"}}
!1 = distinct !{!{!"ejit_entry"}, !{!"ejit_period_arr_ind", !"cell", i32 1}}
