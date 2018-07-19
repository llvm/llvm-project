; RUN: opt -S -functionattrs %s | FileCheck %s

@a = external global i8, !absolute_symbol !0

; CHECK-NOT: define nonnull
define i8* @foo() {
  ret i8* @a
}

!0 = !{i64 0, i64 256}
