; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/weak_resolution.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc

; Verify that prevailing weak for linker symbol is selected across modules,
; non-prevailing ODR are not kept when possible, but non-ODR non-prevailing
; are not affected.
; RUN: llvm-lto -thinlto-action=promote %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=MOD1
; RUN: llvm-lto -thinlto-action=internalize %t.bc -thinlto-index=%t3.bc -exported-symbol=linkoncefunc -o - | llvm-dis -o - | FileCheck %s --check-prefix=MOD1-INT
; RUN: llvm-lto -thinlto-action=promote %t2.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=MOD2
; When exported, we always preserve a linkonce
; RUN: llvm-lto -thinlto-action=promote %t.bc -thinlto-index=%t3.bc -o - --exported-symbol=linkonceodrfuncInSingleModule | llvm-dis -o - | FileCheck %s --check-prefix=EXPORTED

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Alias are resolved, but can't be turned into "available_externally"
; MOD1: @linkonceodralias = weak_odr alias void (), void ()* @linkonceodrfuncwithalias
; MOD2: @linkonceodralias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias
@linkonceodralias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias

; Alias are resolved, but can't be turned into "available_externally"
; MOD1: @linkoncealias = weak alias void (), void ()* @linkoncefuncwithalias
; MOD2: @linkoncealias = linkonce alias void (), void ()* @linkoncefuncwithalias
@linkoncealias = linkonce alias void (), void ()* @linkoncefuncwithalias

; Function with an alias are resolved to weak_odr in prevailing module, but
; not optimized in non-prevailing module (illegal to have an
; available_externally aliasee).
; MOD1: define weak_odr void @linkonceodrfuncwithalias()
; MOD2: define linkonce_odr void @linkonceodrfuncwithalias()
define linkonce_odr void @linkonceodrfuncwithalias() #0 {
entry:
  ret void
}

; Function with an alias are resolved to weak in prevailing module, but
; not optimized in non-prevailing module (illegal to have an
; available_externally aliasee).
; MOD1: define weak void @linkoncefuncwithalias()
; MOD2: define linkonce void @linkoncefuncwithalias()
define linkonce void @linkoncefuncwithalias() #0 {
entry:
  ret void
}

; MOD1: define weak_odr void @linkonceodrfunc()
; MOD2: define available_externally void @linkonceodrfunc()
define linkonce_odr void @linkonceodrfunc() #0 {
entry:
  ret void
}
; MOD1: define weak void @linkoncefunc()
; MOD1-INT: define weak void @linkoncefunc()
; MOD2: declare void @linkoncefunc()
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
; MOD2: declare void @weakfunc()
define weak void @weakfunc() #0 {
entry:
  ret void
}

; MOD1: define weak_odr void @linkonceodrfuncInSingleModule()
; MOD1-INT: define internal void @linkonceodrfuncInSingleModule()
; EXPORTED: define weak_odr void @linkonceodrfuncInSingleModule()
define linkonce_odr void @linkonceodrfuncInSingleModule() #0 {
entry:
  ret void
}
