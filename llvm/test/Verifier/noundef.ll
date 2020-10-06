; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Wrong types for attribute: inalloca nest noalias nocapture noundef nonnull readnone readonly signext sret zeroext byref(void) byval(void) preallocated(void) align 1 dereferenceable(1) dereferenceable_or_null(1)
; CHECK-NEXT: @noundef_void
define noundef void @noundef_void() {
  ret void
}
