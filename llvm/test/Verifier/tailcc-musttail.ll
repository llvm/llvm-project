; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s

declare tailcc void @simple()

define tailcc void @inreg(ptr inreg) {
; CHECK: inreg attribute not allowed in tailcc musttail caller
  musttail call tailcc void @simple()
  ret void
}

define tailcc void @inalloca(ptr inalloca(i8)) {
; CHECK: inalloca attribute not allowed in tailcc musttail caller
  musttail call tailcc void @simple()
  ret void
}

define tailcc void @swifterror(ptr swifterror) {
; CHECK: swifterror attribute not allowed in tailcc musttail caller
  musttail call tailcc void @simple()
  ret void
}

define tailcc void @preallocated(ptr preallocated(i8)) {
; CHECK: preallocated attribute not allowed in tailcc musttail caller
  musttail call tailcc void @simple()
  ret void
}

define tailcc void @byref(ptr byref(i8)) {
; CHECK: byref attribute not allowed in tailcc musttail caller
  musttail call tailcc void @simple()
  ret void
}

define tailcc void @call_inreg() {
; CHECK: inreg attribute not allowed in tailcc musttail callee
  musttail call tailcc void @inreg(ptr inreg undef)
  ret void
}

define tailcc void @call_inalloca() {
; CHECK: inalloca attribute not allowed in tailcc musttail callee
  musttail call tailcc void @inalloca(ptr inalloca(i8) undef)
  ret void
}

define tailcc void @call_swifterror() {
; CHECK: swifterror attribute not allowed in tailcc musttail callee
  %err = alloca swifterror ptr
  musttail call tailcc void @swifterror(ptr swifterror %err)
  ret void
}

define tailcc void @call_preallocated() {
; CHECK: preallocated attribute not allowed in tailcc musttail callee
  musttail call tailcc void @preallocated(ptr preallocated(i8) undef)
  ret void
}

define tailcc void @call_byref() {
; CHECK: byref attribute not allowed in tailcc musttail callee
  musttail call tailcc void @byref(ptr byref(i8) undef)
  ret void
}


declare tailcc void @varargs(...)
define tailcc void @call_varargs(...) {
; CHECK: cannot guarantee tailcc tail call for varargs function
  musttail call tailcc void(...) @varargs(...)
  ret void
}
