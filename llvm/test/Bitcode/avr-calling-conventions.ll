; RUN: llvm-dis < %s.bc | FileCheck %s

; CHECK: define avr_intrcc void @foo(i8 %0)
define avr_intrcc void @foo(i8 %0) {
  ret void
}

; CHECK: define avr_signalcc void @bar(i8 %0)
define avr_signalcc void @bar(i8 %0) {
  ret void
}

; CHECK: define void @baz(i8 %0)
define void @baz(i8 %0) {
  ret void
}
