; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

; CHECK-NOT: @b

@x = constant { ptr, ptr } { ptr @a, ptr @b }
; CHECK: { ptr @a, ptr @a }

define internal void @a() unnamed_addr {
  ret void
}

define internal void @b() unnamed_addr {
  ret void
}
