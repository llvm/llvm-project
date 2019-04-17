; RUN: opt -S -mergefunc < %s | FileCheck %s

; CHECK-NOT: @b

@x = constant { void ()*, void ()* } { void ()* @a, void ()* @b }
; CHECK: { void ()* @a, void ()* @a }

define internal void @a() unnamed_addr {
  ret void
}

define internal void @b() unnamed_addr {
  ret void
}
