; RUN: llvm-dis < %s.bc | FileCheck %s

; CHECK: define avr_intrcc void @foo(i8)
define avr_intrcc void @foo(i8) {
  ret void
}

; CHECK: define avr_signalcc void @bar(i8)
define avr_signalcc void @bar(i8) {
  ret void
}

; CHECK: define void @baz(i8)
define void @baz(i8) {
  ret void
}
