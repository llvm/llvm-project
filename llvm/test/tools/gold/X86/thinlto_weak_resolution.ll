; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_weak_resolution.ll -o %t2.o

; Verify that prevailing weak for linker symbol is kept.
; Note that gold picks the first copy of a function as the prevailing one,
; so listing %t.o first is sufficient to ensure that its copies are prevailing.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=save-temps \
; RUN:     -shared \
; RUN:     -o %t3.o %t.o %t2.o

; RUN: llvm-nm %t3.o | FileCheck %s
; CHECK: weakfunc

; The preempted functions should have been eliminated (the plugin will
; set linkage of odr functions to available_externally, and convert
; linkonce and weak to declarations).
; RUN: llvm-dis %t2.o.4.opt.bc -o - | FileCheck --check-prefix=OPT2 %s
; OPT2: target triple =
; OPT2-NOT: @

; RUN: llvm-dis %t.o.3.import.bc -o - | FileCheck --check-prefix=IMPORT %s
; RUN: llvm-dis %t2.o.3.import.bc -o - | FileCheck --check-prefix=IMPORT2 %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define i32 @main() #0 {
entry:
  call void @linkonceodralias()
  call void @linkoncealias()
  call void @linkonceodrfuncwithalias()
  call void @linkoncefuncwithalias()
  call void @linkonceodrfunc()
  call void @linkoncefunc()
  call void @weakodrfunc()
  call void @weakfunc()
  call void @linkonceodrfuncInSingleModule()
  ret i32 0
}

; Alias are resolved to weak_odr in prevailing module, but left as linkonce_odr
; in non-prevailing module (illegal to have an available_externally alias).
; IMPORT: @linkonceodralias = weak_odr alias void (), void ()* @linkonceodrfuncwithalias
; IMPORT2: @linkonceodralias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias
@linkonceodralias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias

; Alias are resolved in prevailing module, but not optimized in
; non-prevailing module (illegal to have an available_externally alias).
; IMPORT: @linkoncealias = weak alias void (), void ()* @linkoncefuncwithalias
; IMPORT2: @linkoncealias = linkonce alias void (), void ()* @linkoncefuncwithalias
@linkoncealias = linkonce alias void (), void ()* @linkoncefuncwithalias

; Function with an alias are resolved in prevailing module, but
; not optimized in non-prevailing module (illegal to have an
; available_externally aliasee).
; IMPORT: define weak_odr void @linkonceodrfuncwithalias()
; IMPORT2: define linkonce_odr void @linkonceodrfuncwithalias()
define linkonce_odr void @linkonceodrfuncwithalias() #0 {
entry:
  ret void
}

; Function with an alias are resolved to weak in prevailing module, but
; not optimized in non-prevailing module (illegal to have an
; available_externally aliasee).
; IMPORT: define weak void @linkoncefuncwithalias()
; IMPORT2: define linkonce void @linkoncefuncwithalias()
define linkonce void @linkoncefuncwithalias() #0 {
entry:
  ret void
}

; IMPORT: define weak_odr void @linkonceodrfunc()
; IMPORT2: define available_externally void @linkonceodrfunc()
define linkonce_odr void @linkonceodrfunc() #0 {
entry:
  ret void
}
; IMPORT: define weak void @linkoncefunc()
; IMPORT2: declare void @linkoncefunc()
define linkonce void @linkoncefunc() #0 {
entry:
  ret void
}
; IMPORT: define weak_odr void @weakodrfunc()
; IMPORT2: define available_externally void @weakodrfunc()
define weak_odr void @weakodrfunc() #0 {
entry:
  ret void
}
; IMPORT: define weak void @weakfunc()
; IMPORT2: declare void @weakfunc()
define weak void @weakfunc() #0 {
entry:
  ret void
}

; IMPORT: weak_odr void @linkonceodrfuncInSingleModule()
define linkonce_odr void @linkonceodrfuncInSingleModule() #0 {
entry:
  ret void
}
