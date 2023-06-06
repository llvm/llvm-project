;; Test to ensure we properly resolve weak symbols and internalize them when
;; appropriate.

; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/weak_resolution.ll -o %t2.bc

;; First try this with the legacy LTO API
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc
;; Verify that prevailing weak for linker symbol is selected across modules,
;; non-prevailing ODR are not kept when possible, but non-ODR non-prevailing
;; are not affected.
; RUN: llvm-lto -thinlto-action=promote %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=MOD1
; RUN: llvm-lto -thinlto-action=internalize %t.bc -thinlto-index=%t3.bc -exported-symbol=_linkoncefunc -o - | llvm-dis -o - | FileCheck %s --check-prefix=MOD1-INT
; RUN: llvm-lto -thinlto-action=promote %t2.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=MOD2
; When exported, we always preserve a linkonce
; RUN: llvm-lto -thinlto-action=promote %t.bc -thinlto-index=%t3.bc -o - --exported-symbol=_linkonceodrfuncInSingleModule | llvm-dis -o - | FileCheck %s --check-prefix=EXPORTED

;; Now try this with the new LTO API
; RUN: llvm-lto2 run %t.bc %t2.bc -o %t3.out -save-temps \
; RUN:   -r %t.bc,_linkonceodralias,pl \
; RUN:   -r %t.bc,_linkoncealias,pl \
; RUN:   -r %t.bc,_linkonceodrvarInSingleModule,pl \
; RUN:   -r %t.bc,_weakodrvarInSingleModule,pl \
; RUN:   -r %t.bc,_linkonceodrfuncwithalias,pl \
; RUN:   -r %t.bc,_linkoncefuncwithalias,pl \
; RUN:   -r %t.bc,_linkonceodrfunc,pl \
; RUN:   -r %t.bc,_linkoncefunc,pl \
; RUN:   -r %t.bc,_weakodrfunc,pl \
; RUN:   -r %t.bc,_weakfunc,pl \
; RUN:   -r %t.bc,_linkonceodrfuncInSingleModule,pl \
; RUN:   -r %t2.bc,_linkonceodrfuncwithalias,l \
; RUN:   -r %t2.bc,_linkoncefuncwithalias,l \
; RUN:   -r %t2.bc,_linkonceodrfunc,l \
; RUN:   -r %t2.bc,_linkoncefunc,l \
; RUN:   -r %t2.bc,_weakodrfunc,l \
; RUN:   -r %t2.bc,_weakfunc,l \
; RUN:   -r %t2.bc,_linkonceodralias,l \
; RUN:   -r %t2.bc,_linkoncealias,l
; RUN: llvm-dis %t3.out.1.2.internalize.bc -o - | FileCheck %s --check-prefix=MOD1-INT

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

;; Alias are resolved, but can't be turned into "available_externally"
; MOD1: @linkonceodralias = weak_odr alias void (), ptr @linkonceodrfuncwithalias
; MOD2: @linkonceodralias = linkonce_odr alias void (), ptr @linkonceodrfuncwithalias
@linkonceodralias = linkonce_odr alias void (), ptr @linkonceodrfuncwithalias

;; Alias are resolved, but can't be turned into "available_externally"
; MOD1: @linkoncealias = weak alias void (), ptr @linkoncefuncwithalias
; MOD2: @linkoncealias = linkonce alias void (), ptr @linkoncefuncwithalias
@linkoncealias = linkonce alias void (), ptr @linkoncefuncwithalias

;; Non-exported linkonce/weak variables can always be internalized, regardless
;; of whether they are const or *unnamed_addr.
; MOD1-INT: @linkonceodrvarInSingleModule = internal global
; MOD1-INT: @weakodrvarInSingleModule = internal global
@linkonceodrvarInSingleModule = linkonce_odr dso_local global ptr null, align 8
@weakodrvarInSingleModule = weak_odr dso_local global ptr null, align 8

;; Function with an alias are resolved to weak_odr in prevailing module, but
;; not optimized in non-prevailing module (illegal to have an
;; available_externally aliasee).
; MOD1: define weak_odr void @linkonceodrfuncwithalias()
; MOD2: define linkonce_odr void @linkonceodrfuncwithalias()
define linkonce_odr void @linkonceodrfuncwithalias() #0 {
entry:
  ret void
}

;; Function with an alias are resolved to weak in prevailing module, but
;; not optimized in non-prevailing module (illegal to have an
;; available_externally aliasee).
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
;; New LTO API will use dso_local
; MOD1-INT: define weak{{.*}} void @linkoncefunc()
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

;; A linkonce_odr with a single, non-exported, def can be safely
;; internalized without increasing code size or being concerned
;; about affecting function pointer equality.
; MOD1: define weak_odr void @linkonceodrfuncInSingleModule()
; MOD1-INT: define internal void @linkonceodrfuncInSingleModule()
; EXPORTED: define weak_odr void @linkonceodrfuncInSingleModule()
define linkonce_odr void @linkonceodrfuncInSingleModule() #0 {
entry:
  ret void
}
