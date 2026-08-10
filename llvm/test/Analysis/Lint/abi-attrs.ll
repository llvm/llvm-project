; RUN: opt < %s -passes=lint -disable-output 2>&1 | FileCheck %s

declare void @fn_nothing_i8(i8 %x)
declare void @fn_zeroext(i8 zeroext %x)
declare void @fn_signext(i8 signext %x)
declare void @fn_inreg(i8 inreg %x)

declare void @fn_nothing_ptr(ptr %x)
declare void @fn_byval(ptr byval(i8) %x)
declare void @fn_byref(ptr byref(i8) %x)
declare void @fn_inalloca(ptr inalloca(i8) %x)
declare void @fn_preallocated(ptr preallocated(i8) %x)
declare void @fn_sret(ptr sret(i8) %x)

define void @caller_zeroext(i8 %x) {
; CHECK: Undefined behavior: ABI attribute zeroext not present on both function and call-site
; CHECK:  call void @fn_zeroext(i8 %x)
  call void @fn_zeroext(i8 %x)

; CHECK: Undefined behavior: ABI attribute zeroext not present on both function and call-site
; CHECK:  call void @fn_nothing_i8(i8 zeroext %x)
  call void @fn_nothing_i8(i8 zeroext %x)
  ret void
}

define void @caller_signext(i8 %x) {
; CHECK: Undefined behavior: ABI attribute signext not present on both function and call-site
; CHECK:  call void @fn_signext(i8 %x)
  call void @fn_signext(i8 %x)

; CHECK: Undefined behavior: ABI attribute signext not present on both function and call-site
; CHECK:  call void @fn_nothing_i8(i8 signext %x)
  call void @fn_nothing_i8(i8 signext %x)
  ret void
}

define void @caller_inreg(i8 %x) {
; CHECK: Undefined behavior: ABI attribute inreg not present on both function and call-site
; CHECK:  call void @fn_inreg(i8 %x)
  call void @fn_inreg(i8 %x)

; CHECK: Undefined behavior: ABI attribute inreg not present on both function and call-site
; CHECK:  call void @fn_nothing_i8(i8 inreg %x)
  call void @fn_nothing_i8(i8 inreg %x)
  ret void
}

define void @caller_byval(ptr %x) {
; CHECK: Undefined behavior: ABI attribute byval not present on both function and call-site
; CHECK:  call void @fn_byval(ptr %x)
  call void @fn_byval(ptr %x)

; CHECK: Undefined behavior: ABI attribute byval not present on both function and call-site
; CHECK:  call void @fn_nothing_ptr(ptr byval(i8) %x)
  call void @fn_nothing_ptr(ptr byval(i8) %x)

; CHECK: Undefined behavior: ABI attribute byval does not have same argument for function and call-site
; CHECK:  call void @fn_byval(ptr byval(i16) %x)
  call void @fn_byval(ptr byval(i16) %x)
  ret void
}

define void @caller_byref(ptr %x) {
; CHECK: Undefined behavior: ABI attribute byref not present on both function and call-site
; CHECK:  call void @fn_byref(ptr %x)
  call void @fn_byref(ptr %x)

; CHECK: Undefined behavior: ABI attribute byref not present on both function and call-site
; CHECK:  call void @fn_nothing_ptr(ptr byref(i8) %x)
  call void @fn_nothing_ptr(ptr byref(i8) %x)

; CHECK: Undefined behavior: ABI attribute byref does not have same argument for function and call-site
; CHECK:  call void @fn_byref(ptr byref(i16) %x)
  call void @fn_byref(ptr byref(i16) %x)
  ret void
}

define void @caller_inalloca(ptr %x) {
; CHECK: Undefined behavior: ABI attribute inalloca not present on both function and call-site
; CHECK:  call void @fn_inalloca(ptr %x)
  call void @fn_inalloca(ptr %x)

; CHECK: Undefined behavior: ABI attribute inalloca not present on both function and call-site
; CHECK:  call void @fn_nothing_ptr(ptr inalloca(i8) %x)
  call void @fn_nothing_ptr(ptr inalloca(i8) %x)

; CHECK: Undefined behavior: ABI attribute inalloca does not have same argument for function and call-site
; CHECK:  call void @fn_inalloca(ptr inalloca(i16) %x)
  call void @fn_inalloca(ptr inalloca(i16) %x)
  ret void
}

define void @caller_sret(ptr %x) {
; CHECK: Undefined behavior: ABI attribute sret not present on both function and call-site
; CHECK:  call void @fn_sret(ptr %x)
  call void @fn_sret(ptr %x)

; CHECK: Undefined behavior: ABI attribute sret not present on both function and call-site
; CHECK:  call void @fn_nothing_ptr(ptr sret(i8) %x)
  call void @fn_nothing_ptr(ptr sret(i8) %x)

; CHECK: Undefined behavior: ABI attribute sret does not have same argument for function and call-site
; CHECK:  call void @fn_sret(ptr sret(i16) %x)
  call void @fn_sret(ptr sret(i16) %x)
  ret void
}
