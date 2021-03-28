; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Wrong types for attribute: nest noalias nocapture noundef nonnull readnone readonly signext zeroext byref(void) byval(void) inalloca(void) preallocated(void) sret(void) align 1 dereferenceable(1) dereferenceable_or_null(1)
; CHECK-NEXT: @noundef_void
define noundef void @noundef_void() {
  ret void
}
