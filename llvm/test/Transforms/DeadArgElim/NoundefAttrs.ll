; RUN: opt -deadargelim -S < %s | FileCheck %s

; If caller is changed to pass in undef, noundef and related attributes
; should be deleted.


; CHECK:   define i64 @bar(i64* %0, i64 %1)
define i64 @bar(i64* nonnull dereferenceable(8) %0, i64 %1) {
entry:
  %2 = add i64 %1, 8
  ret i64 %2
}

define i64 @foo(i64* %p, i64 %v) {
; CHECK:   %retval = call i64 @bar(i64* undef, i64 %v)
  %retval = call i64 @bar(i64* nonnull dereferenceable(8) %p, i64 %v)
  ret i64 %retval
}
