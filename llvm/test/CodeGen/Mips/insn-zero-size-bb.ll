; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips | FileCheck %s
; RUN: llc < %s -march=mips -mattr=mips16 | FileCheck %s

; Verify that we emit the .insn directive for zero-sized (empty) basic blocks.
; This only really matters for microMIPS and MIPS16.

declare i32 @foo(...)
declare void @bar()

define void @main() personality i8* bitcast (i32 (...)* @foo to i8*) {
entry:
  invoke void @bar() #0
          to label %unreachable unwind label %return

unreachable:
; CHECK:          {{.*}}: # %unreachable
; CHECK-NEXT:         .insn
  unreachable

return:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  ret void
}

attributes #0 = { noreturn }
