; RUN: not opt -verify %s 2>&1 | FileCheck %s

declare swifttailcc void @simple()

define swifttailcc void @inreg(i8* inreg) {
; CHECK: inreg attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @inalloca(i8* inalloca(i8)) {
; CHECK: inalloca attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @swifterror(i8** swifterror) {
; CHECK: swifterror attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @preallocated(i8* preallocated(i8)) {
; CHECK: preallocated attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @byref(i8* byref(i8)) {
; CHECK: byref attribute not allowed in swifttailcc musttail caller
  musttail call swifttailcc void @simple()
  ret void
}

define swifttailcc void @call_inreg() {
; CHECK: inreg attribute not allowed in swifttailcc musttail callee
  musttail call swifttailcc void @inreg(i8* inreg undef)
  ret void
}

define swifttailcc void @call_inalloca() {
; CHECK: inalloca attribute not allowed in swifttailcc musttail callee
  musttail call swifttailcc void @inalloca(i8* inalloca(i8) undef)
  ret void
}

define swifttailcc void @call_swifterror() {
; CHECK: swifterror attribute not allowed in swifttailcc musttail callee
  %err = alloca swifterror i8*
  musttail call swifttailcc void @swifterror(i8** swifterror %err)
  ret void
}

define swifttailcc void @call_preallocated() {
; CHECK: preallocated attribute not allowed in swifttailcc musttail callee
  musttail call swifttailcc void @preallocated(i8* preallocated(i8) undef)
  ret void
}

define swifttailcc void @call_byref() {
; CHECK: byref attribute not allowed in swifttailcc musttail callee
  musttail call swifttailcc void @byref(i8* byref(i8) undef)
  ret void
}


declare swifttailcc void @varargs(...)
define swifttailcc void @call_varargs(...) {
; CHECK: cannot guarantee swifttailcc tail call for varargs function
  musttail call swifttailcc void(...) @varargs(...)
  ret void
}
