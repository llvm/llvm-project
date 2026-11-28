; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; optnone implies noinline, so noinline is not required alongside it.
; CHECK: define void @optnone_only() #[[OPTNONE:[0-9]+]]
define void @optnone_only() optnone {
  ret void
}

; CHECK: define void @optnone_noinline() #[[NOINLINE_OPTNONE:[0-9]+]]
define void @optnone_noinline() noinline optnone {
  ret void
}

; alwaysinline may be combined with optnone; alwaysinline takes precedence over
; the noinline implied by optnone.
; CHECK: define void @optnone_alwaysinline() #[[ALWAYSINLINE_OPTNONE:[0-9]+]]
define void @optnone_alwaysinline() alwaysinline optnone {
  ret void
}

; CHECK-DAG: attributes #[[OPTNONE]] = { optnone }
; CHECK-DAG: attributes #[[NOINLINE_OPTNONE]] = { noinline optnone }
; CHECK-DAG: attributes #[[ALWAYSINLINE_OPTNONE]] = { alwaysinline optnone }
