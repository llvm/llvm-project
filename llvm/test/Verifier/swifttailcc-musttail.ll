; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s

declare swifttailcc void @simple()

define swifttailcc void @inreg(ptr inreg) {
; CHECK: inreg attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @inalloca(ptr inalloca(i8)) {
; CHECK: inalloca attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @swifterror(ptr swifterror) {
; CHECK: swifterror attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @preallocated(ptr preallocated(i8)) {
; CHECK: preallocated attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @byref(ptr byref(i8)) {
; CHECK: byref attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @call_inreg() {
; CHECK: inreg attribute not allowed in swifttailcc musttail callee
  musttail call swifttailcc void @inreg(ptr inreg undef)
  ret void
}

define swifttailcc void @call_inalloca() {
; CHECK: inalloca attribute not allowed in swifttailcc musttail callee
  musttail call swifttailcc void @inalloca(ptr inalloca(i8) undef)
  ret void
}

define swifttailcc void @call_swifterror() {
; CHECK: swifterror attribute not allowed in swifttailcc musttail callee
  %err = alloca swifterror ptr
  musttail call swifttailcc void @swifterror(ptr swifterror %err)
  ret void
}

define swifttailcc void @call_preallocated() {
; CHECK: preallocated attribute not allowed in swifttailcc musttail callee
  musttail call swifttailcc void @preallocated(ptr preallocated(i8) undef)
  ret void
}

define swifttailcc void @call_byref() {
; CHECK: byref attribute not allowed in swifttailcc musttail callee
  musttail call swifttailcc void @byref(ptr byref(i8) undef)
  ret void
}


declare swifttailcc void @varargs(...)
define swifttailcc void @call_varargs(...) {
; CHECK: cannot guarantee swifttailcc tail call for varargs function
  musttail call swifttailcc void(...) @varargs(...)
  ret void
}
