; Do setup work for all below tests: generate bitcode and combined index
; RUN: llvm-as -module-summary %s -o %t.bc
; RUN: llvm-as -module-summary %p/Inputs/odr_resolution.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc

; Verify that only one ODR is selected across modules, but non ODR are not affected.
; RUN: llvm-lto -thinlto-action=promote %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=MOD1
; RUN: llvm-lto -thinlto-action=promote %t2.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=MOD2

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Alias are not optimized
; MOD1: @linkoncealias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias
; MOD2: @linkoncealias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias
@linkoncealias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias

; Function with an alias are not optimized
; MOD1: define linkonce_odr void @linkonceodrfuncwithalias()
; MOD2: define linkonce_odr void @linkonceodrfuncwithalias()
define linkonce_odr void @linkonceodrfuncwithalias() #0 {
entry:
  ret void
}

; MOD1: define weak_odr void @linkonceodrfunc()
; MOD2: define available_externally void @linkonceodrfunc()
define linkonce_odr void @linkonceodrfunc() #0 {
entry:
  ret void
}
; MOD1: define linkonce void @linkoncefunc()
; MOD2: define linkonce void @linkoncefunc()
define linkonce void @linkoncefunc() #0 {
entry:
  ret void
}
; MOD1: define weak_odr void @weakodrfunc()
; MOD2: define available_externally void @weakodrfunc()
define weak_odr void @weakodrfunc() #0 {
entry:
  ret void
}
; MOD1: define weak void @weakfunc()
; MOD2: define weak void @weakfunc()
define weak void @weakfunc() #0 {
entry:
  ret void
}

