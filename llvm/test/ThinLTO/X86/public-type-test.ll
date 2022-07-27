; Test to ensure that the legacy LTO API lowers @llvm.public.type.test.

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto --thinlto-action=run -exported-symbol=_main %t.bc --thinlto-save-temps=%t2
; RUN: llvm-dis -o - %t20.2.internalized.bc | FileCheck %s --check-prefix=PUBLIC
; RUN: llvm-lto --thinlto-action=run -exported-symbol=_main %t.bc --thinlto-save-temps=%t2 --whole-program-visibility
; RUN: llvm-dis -o - %t20.2.internalized.bc | FileCheck %s --check-prefix=HIDDEN

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
