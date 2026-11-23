; RUN: not llc -mtriple=aarch64-linux-gnu < %s 2>&1 | FileCheck %s

; An aggregate that is not 1, 8, 16, 32, 64 or 128 bits wide has no simple
; value type (MVT::Other). Check that using it with a register constraint is
; diagnosed instead of crashing in getRegForInlineAsmConstraint.

%struct.S = type { i64, i64, i64 }

; CHECK: error: cannot handle indirect register inputs yet for constraint 'r'
define void @input_r(ptr %p) {
  call void asm sideeffect "", "*r"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: cannot handle indirect register inputs yet for constraint 'x'
define void @input_x(ptr %p) {
  call void asm sideeffect "", "*x"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: cannot handle indirect register inputs yet for constraint 'w'
define void @input_w(ptr %p) {
  call void asm sideeffect "", "*w"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: could not allocate output register for constraint 'r'
define void @output_r(ptr %p) {
  call void asm sideeffect "", "=*r"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: could not allocate output register for constraint 'x'
define void @output_x(ptr %p) {
  call void asm sideeffect "", "=*x"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: could not allocate output register for constraint 'w'
define void @output_w(ptr %p) {
  call void asm sideeffect "", "=*w"(ptr elementtype(%struct.S) %p)
  ret void
}
