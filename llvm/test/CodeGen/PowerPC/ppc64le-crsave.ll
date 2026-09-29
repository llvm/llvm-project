; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@_ZTIi = external constant ptr
declare ptr @__cxa_allocate_exception(i64)
declare void @__cxa_throw(ptr, ptr, ptr)

define void @crsave() {
entry:
  call void asm sideeffect "", "~{cr2}"()
  call void asm sideeffect "", "~{cr3}"()
  call void asm sideeffect "", "~{cr4}"()

  %exception = call ptr @__cxa_allocate_exception(i64 4)
  store i32 0, ptr %exception
  call void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null)
  unreachable

return:                                           ; No predecessors!
  ret void
}
; CHECK-LABEL: @crsave
; CHECK: .cfi_offset cr2, 8
; CHECK: .cfi_offset cr3, 8
; CHECK: .cfi_offset cr4, 8

