; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; optnone implies noinline, so noinline is not required alongside it.
; CHECK-NOT: optnone_only
define void @optnone_only() optnone {
  ret void
}

; CHECK-NOT: optnone_noinline
define void @optnone_noinline() noinline optnone {
  ret void
}

; CHECK: Attributes 'alwaysinline and optnone' are incompatible!
define void @optnone_alwaysinline() alwaysinline optnone {
  ret void
}
