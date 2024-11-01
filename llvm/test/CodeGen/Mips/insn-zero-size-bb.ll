; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips | FileCheck %s
; RUN: llc < %s -march=mips -mattr=mips16 | FileCheck %s

; Verify that we emit the .insn directive for zero-sized (empty) basic blocks.
; This only really matters for microMIPS and MIPS16.

declare i32 @foo(...)
declare void @bar()

define void @main() personality ptr @foo {
entry:
  invoke void @bar() #0
          to label %unreachable unwind label %return

unreachable:
; CHECK:          {{.*}}: # %unreachable
; CHECK-NEXT:         .insn
  unreachable

return:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret void
}

attributes #0 = { noreturn }
