; RUN: llc -mtriple=thumbv7-windows -mcpu=cortex-a9 -o - %s | FileCheck %s

define void @function() nounwind {
entry:
  ret void
}

; CHECK-LABEL: function

