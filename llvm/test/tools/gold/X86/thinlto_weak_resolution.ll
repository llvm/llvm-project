; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_weak_resolution.ll -o %t2.o

; Verify that prevailing weak for linker symbol is kept.
; Note that gold picks the first copy of a function as the prevailing one,
; so listing %t.o first is sufficient to ensure that its copies are prevailing.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=save-temps \
; RUN:     -shared \
; RUN:     -o %t3.o %t.o %t2.o

; RUN: llvm-nm %t3.o | FileCheck %s
; CHECK: weakfunc

; All of the preempted functions should have been eliminated (the plugin will
; not link them in).
; RUN: llvm-dis %t2.o.opt.bc -o - | FileCheck --check-prefix=OPT2 %s
; OPT2-NOT: @

; RUN: llvm-dis %t.o.opt.bc -o - | FileCheck --check-prefix=OPT %s

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

; Alias are resolved
; OPT: @linkonceodralias = weak_odr alias void (), void ()* @linkonceodrfuncwithalias
@linkonceodralias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias

; Alias are resolved
; OPT: @linkoncealias = weak alias void (), void ()* @linkoncefuncwithalias
@linkoncealias = linkonce alias void (), void ()* @linkoncefuncwithalias

; Function with an alias are not optimized
; OPT: define linkonce_odr void @linkonceodrfuncwithalias()
define linkonce_odr void @linkonceodrfuncwithalias() #0 {
entry:
  ret void
}

; Function with an alias are not optimized
; OPT: define linkonce void @linkoncefuncwithalias()
define linkonce void @linkoncefuncwithalias() #0 {
entry:
  ret void
}

; OPT: define weak_odr void @linkonceodrfunc()
define linkonce_odr void @linkonceodrfunc() #0 {
entry:
  ret void
}
; OPT: define weak void @linkoncefunc()
define linkonce void @linkoncefunc() #0 {
entry:
  ret void
}
; OPT: define weak_odr void @weakodrfunc()
define weak_odr void @weakodrfunc() #0 {
entry:
  ret void
}
; OPT: define weak void @weakfunc()
define weak void @weakfunc() #0 {
entry:
  ret void
}

; OPT: weak_odr void @linkonceodrfuncInSingleModule()
define linkonce_odr void @linkonceodrfuncInSingleModule() #0 {
entry:
  ret void
}
