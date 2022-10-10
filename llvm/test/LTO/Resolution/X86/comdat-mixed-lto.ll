; Test of comdat handling with mixed thinlto and regular lto compilation.

; This module is compiled with ThinLTO
; RUN: opt -module-summary -o %t1.o %s
; Input module compiled for regular LTO
; RUN: opt -o %t2.o %p/Inputs/comdat-mixed-lto.ll

; The copy of C from this module is prevailing. The copy of C from the
; regular LTO module is not prevailing, and will be dropped to
; available_externally.
; RUN: llvm-lto2 run -r=%t1.o,C,pl -r=%t2.o,C,l -r=%t1.o,testglobfunc,lxp -r=%t2.o,testglobfunc,lx -o %t3 %t1.o %t2.o -save-temps

; The Input module (regular LTO) is %t3.0. Check to make sure that we removed
; __cxx_global_var_init and testglobfunc from comdat. Also check to ensure
; that testglobfunc was dropped to available_externally. Otherwise we would
; have linker multiply defined errors as it is no longer in a comdat and
; would clash with the copy from this module.
; RUN: llvm-dis %t3.0.0.preopt.bc -o - | FileCheck %s

; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init, ptr @C }]
; CHECK: @C = available_externally dso_local global %"class.Test::ptr" zeroinitializer, align 4
; CHECK-NOT: declare
; CHECK: declare dso_local void @__cxx_global_var_init() section ".text.startup"
; CHECK-NOT: declare

; ModuleID = 'comdat-mixed-lto.o'
source_filename = "comdat-mixed-lto.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.Test::ptr" = type { i32 }

$C = comdat any

@C = linkonce_odr global %"class.Test::ptr" zeroinitializer, comdat, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @__cxx_global_var_init, i8* bitcast (%"class.Test::ptr"* @C to i8*) }]
define void @testglobfunc() #1 section ".text.startup" comdat($C) {
entry:
  ret void
}

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #1 section ".text.startup" comdat($C) {
entry:
  ret void
}
