; RUN: llvm-as < %s | llvm-dis | FileCheck --strict-whitespace %s

; CHECK: declare avr_intrcc void @intr()
declare avr_intrcc void @intr()

; CHECK: declare avr_signalcc void @signal()
declare avr_signalcc void @signal()

; CHECK: declare avr_builtincc void @builtin()
declare avr_builtincc void @builtin()
