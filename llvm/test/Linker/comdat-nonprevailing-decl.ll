; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link -S %t/1.ll %t/1-aux.ll -o - | FileCheck %s
; RUN: llvm-link -S %t/2.ll %t/2-aux.ll -o - | FileCheck %s --check-prefix=CHECK2

;--- 1.ll
$c = comdat any

@v = global i32 0, comdat ($c)

; CHECK: @v = global i32 0, comdat($c)
; CHECK: @v3 = external global i32
; CHECK: @v2 = external dllexport global i32

;--- 1-aux.ll
$c = comdat any

@v2 = weak dllexport global i32 0, comdat ($c)
define ptr @f2() {
  ret ptr @v2
}

@v3 = weak alias i32, ptr @v2
define ptr @f3() {
  ret ptr @v3
}

;--- 2.ll
;; Check that a private global variable from a non-prevailing comdat group is
;; converted into 'available_externally' and excluded from the comdat group.

; CHECK2: $__profc_foo = comdat any
; CHECK2: @llvm.compiler.used = appending global [2 x ptr] [ptr @__profd_foo.[[SUFFIX:[0-9]+]], ptr @__profd_foo]
; CHECK2: @__profd_foo.[[SUFFIX]] = private global ptr @__profc_foo, comdat($__profc_foo)
; CHECK2: @__profc_foo = linkonce_odr global i64 1, comdat
; CHECK2: @__profd_foo = available_externally dso_local global ptr @__profc_foo{{$}}

$__profc_foo = comdat any
@__profc_foo = linkonce_odr global i64 1, comdat
@__profd_foo = private global ptr @__profc_foo, comdat($__profc_foo)
@llvm.compiler.used = appending global [1 x ptr] [ ptr @__profd_foo ]

define ptr @bar() {
  ret ptr @__profc_foo
}

;--- 2-aux.ll
$__profc_foo = comdat any
@__profc_foo = linkonce_odr global i64 1, comdat
@__profd_foo = private global ptr @__profc_foo, comdat($__profc_foo)
@llvm.compiler.used = appending global [1 x ptr] [ ptr @__profd_foo ]

define ptr @baz() {
  ret ptr @__profc_foo
}
