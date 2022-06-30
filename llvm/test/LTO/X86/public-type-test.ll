; Test to ensure that the LTO API (legacy and new) lowers @llvm.public.type.test.

; RUN: llvm-as < %s > %t1
; RUN: llvm-lto -exported-symbol=_main %t1 -o %t2 --lto-save-before-opt --whole-program-visibility
; RUN: llvm-dis -o - %t2.0.preopt.bc | FileCheck %s --check-prefix=HIDDEN
; RUN: llvm-lto -exported-symbol=_main %t1 -o %t2 --lto-save-before-opt
; RUN: llvm-dis -o - %t2.0.preopt.bc | FileCheck %s --check-prefix=PUBLIC

; RUN: llvm-lto2 run %t1 -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t2 \
; RUN:   -r=%t1,_main,px
; RUN: llvm-dis %t2.0.0.preopt.bc -o - | FileCheck %s --check-prefix=HIDDEN
; RUN: llvm-lto2 run %t1 -save-temps -pass-remarks=. \
; RUN:   -o %t2 \
; RUN:   -r=%t1,_main,px
; RUN: llvm-dis %t2.0.0.preopt.bc -o - | FileCheck %s --check-prefix=PUBLIC

; PUBLIC-NOT: call {{.*}}@llvm.public.type.test
; PUBLIC-NOT: call {{.*}}@llvm.type.test
; HIDDEN-NOT: call {{.*}}@llvm.public.type.test
; HIDDEN: call {{.*}}@llvm.type.test

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

define i32 @main(ptr %vtable) {
entry:
  %p = call i1 @llvm.public.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  ret i32 0
}

declare void @llvm.assume(i1)
declare i1 @llvm.public.type.test(ptr, metadata)
