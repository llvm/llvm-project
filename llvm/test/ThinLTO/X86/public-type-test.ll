; Test to ensure that the LTO API (legacy and new) lowers @llvm.public.type.test.

; RUN: split-file %s %t

; RUN: opt -module-summary %t/main.ll -o %t/main.bc
; RUN: opt -module-summary %t/foo.ll -o %t/foo.bc
; RUN: llvm-lto --thinlto-action=run -exported-symbol=_main %t/main.bc %t/foo.bc --thinlto-save-temps=%t2.
; RUN: llvm-dis -o - %t2.0.3.imported.bc | FileCheck %s --check-prefix=PUBLIC
; RUN: llvm-lto --thinlto-action=run -exported-symbol=_main %t/main.bc %t/foo.bc --thinlto-save-temps=%t2. --whole-program-visibility
; RUN: llvm-dis -o - %t2.0.3.imported.bc | FileCheck %s --check-prefix=HIDDEN

; RUN: llvm-lto2 run %t/main.bc %t/foo.bc -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:   -o %t3 \
; RUN:   -r=%t/main.bc,_main,px \
; RUN:   -r=%t/main.bc,_bar,px \
; RUN:   -r=%t/main.bc,_foo, \
; RUN:   -r=%t/foo.bc,_foo,px
; RUN: llvm-dis %t3.1.3.import.bc -o - | FileCheck %s --check-prefix=HIDDEN
; RUN: llvm-lto2 run %t/main.bc %t/foo.bc -save-temps -pass-remarks=. \
; RUN:   -o %t3 \
; RUN:   -r=%t/main.bc,_main,px \
; RUN:   -r=%t/main.bc,_bar,px \
; RUN:   -r=%t/main.bc,_foo, \
; RUN:   -r=%t/foo.bc,_foo,px
; RUN: llvm-dis %t3.1.3.import.bc -o - | FileCheck %s --check-prefix=PUBLIC

; PUBLIC-NOT: call {{.*}}@llvm.public.type.test
; PUBLIC-NOT: call {{.*}}@llvm.type.test
;; We should have converted the type tests from both main and the imported
;; copy of foo to non-public.
; HIDDEN-NOT: call {{.*}}@llvm.public.type.test
; HIDDEN: call {{.*}}@llvm.type.test
; HIDDEN-NOT: call {{.*}}@llvm.public.type.test
; HIDDEN: call {{.*}}@llvm.type.test

;--- main.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

define i32 @main(ptr %vtable) {
entry:
  %p = call i1 @llvm.public.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  call void @bar(ptr %vtable)
  ret i32 0
}

define void @bar(ptr %vtable) {
entry:
  call void @foo(ptr %vtable)
  ret void
}

declare void @foo(ptr %vtable)

declare void @llvm.assume(i1)
declare i1 @llvm.public.type.test(ptr, metadata)

;--- foo.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

define void @foo(ptr %vtable) {
entry:
  %p = call i1 @llvm.public.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  ret void
}

declare void @llvm.assume(i1)
declare i1 @llvm.public.type.test(ptr, metadata)
