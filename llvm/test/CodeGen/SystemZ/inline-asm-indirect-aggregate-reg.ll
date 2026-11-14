; RUN: not llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 2>&1 | FileCheck %s

; An aggregate that is not 1, 8, 16, 32, 64 or 128 bits wide has no simple
; value type (MVT::Other). Check that using it with a register constraint is
; diagnosed instead of crashing in getRegForInlineAsmConstraint.

%struct.S = type { i64, i64, i64 }

; CHECK: error: cannot handle indirect register inputs yet for constraint 'r'
define void @input_r(ptr %p) {
  call void asm sideeffect "", "*r"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: cannot handle indirect register inputs yet for constraint 'd'
define void @input_d(ptr %p) {
  call void asm sideeffect "", "*d"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: cannot handle indirect register inputs yet for constraint 'f'
define void @input_f(ptr %p) {
  call void asm sideeffect "", "*f"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: cannot handle indirect register inputs yet for constraint 'v'
define void @input_v(ptr %p) {
  call void asm sideeffect "", "*v"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: could not allocate output register for constraint 'r'
define void @output_r(ptr %p) {
  call void asm sideeffect "", "=*r"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: could not allocate output register for constraint 'd'
define void @output_d(ptr %p) {
  call void asm sideeffect "", "=*d"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: could not allocate output register for constraint 'f'
define void @output_f(ptr %p) {
  call void asm sideeffect "", "=*f"(ptr elementtype(%struct.S) %p)
  ret void
}

; CHECK: error: could not allocate output register for constraint 'v'
define void @output_v(ptr %p) {
  call void asm sideeffect "", "=*v"(ptr elementtype(%struct.S) %p)
  ret void
}
