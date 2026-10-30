; RUN: llc < %s -mtriple=nvptx64 -debug-only=isel 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; An i1 load is custom-lowered to a zext load to i16 plus a truncate.
; LegalizeLoadOps replaces the original load's chain result with the chain the
; lowering returns, so the lowering must return the new load's chain: returning
; the original one leaves the new load unordered with respect to everything
; that followed the original load. The store to %p aliases the load, and its
; stored value does not come from the load, so only the chain orders them.

target triple = "nvptx64-nvidia-cuda"

define void @foo(ptr %p, ptr %q) {
; CHECK: [[LD:t[0-9]+]]: i16,ch = load<{{.*}}load (s8) from %ir.p{{.*}}
; CHECK: store<(store (s8) into %ir.p), trunc to i8> [[LD]]:1,
  %v = load i1, ptr %p
  store i1 true, ptr %p
  store i1 %v, ptr %q
  ret void
}
