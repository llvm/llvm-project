; RUN: rm -rf %t && split-file %s %t && cd %t

;--- const.ll

; RUN: opt -S < const.ll | FileCheck %s --check-prefix=CONST

@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo1, i32 0, i64 55764), ptr null }, { i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo2, i32 0, i64 55764), ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @bar, i32 0, i64 55764), ptr null }]

define void @foo1() {
  ret void
}

define void @foo2() {
  ret void
}

define void @bar() {
  ret void
}

; CONST: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo1, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @foo2, ptr null }]
; CONST: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar, ptr null }]
; CONST: !llvm.module.flags = !{!0, !1}
; CONST: !0 = !{i32 1, !"ptrauth-init-fini", i32 1}
; CONST: !1 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 0}

;--- blended.ll

; RUN: opt -S < blended.ll | FileCheck %s --check-prefix=BLENDED

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), ptr null }]
@llvm.global_dtors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @bar1, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), ptr null }, { i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @bar2, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), ptr null }]

define void @foo() {
  ret void
}

define void @bar1() {
  ret void
}

define void @bar2() {
  ret void
}

; BLENDED: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]
; BLENDED: @llvm.global_dtors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar1, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @bar2, ptr null }]
; BLENDED: !llvm.module.flags = !{!0, !1}
; BLENDED: !0 = !{i32 1, !"ptrauth-init-fini", i32 1}
; BLENDED: !1 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 1}

;--- err1.ll

; RUN: not opt -S < err1.ll 2>&1 | FileCheck %s --check-prefix=ERR1

; ERR1: signing of ctors/dtors should be requested via module flags

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764, ptr inttoptr (i64 2 to ptr)), ptr null }]

define void @foo() {
  ret void
}

;--- err2.ll

; RUN: not opt -S < err2.ll 2>&1 | FileCheck %s --check-prefix=ERR2

; ERR2: signing of ctors/dtors should be requested via module flags

@g = external global ptr
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @bar, i32 0, i64 55764, ptr @g), ptr null }]

define void @bar() {
  ret void
}

;--- disagreement1.ll

; RUN: not opt -S < disagreement1.ll 2>&1 | FileCheck %s --check-prefix=DISAGREEMENT1

; DISAGREEMENT1: signing of ctors/dtors should be requested via module flags

@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764), ptr null }, { i32, ptr, ptr } { i32 65535, ptr @bar, ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

;--- disagreement2.ll

; RUN: not opt -S < disagreement2.ll 2>&1 | FileCheck %s --check-prefix=DISAGREEMENT2

; DISAGREEMENT2: signing of ctors/dtors should be requested via module flags

@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764), ptr null }, { i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @bar, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

;--- disagreement3.ll

; RUN: not opt -S < disagreement3.ll 2>&1 | FileCheck %s --check-prefix=DISAGREEMENT3

; DISAGREEMENT3: signing of ctors/dtors should be requested via module flags

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764), ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @bar, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

;--- existing-flags-1.ll

; RUN: not opt -S < existing-flags-1.ll 2>&1 | FileCheck %s --check-prefix=EXISTING-FLAGS-1

; EXISTING-FLAGS-1: signing of ctors/dtors should be requested via module flags

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764), ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ptrauth-init-fini", i32 1}

;--- existing-flags-2.ll

; RUN: not opt -S < existing-flags-2.ll 2>&1 | FileCheck %s --check-prefix=EXISTING-FLAGS-2

; EXISTING-FLAGS-2: signing of ctors/dtors should be requested via module flags

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr ptrauth (ptr @foo, i32 0, i64 55764), ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ptrauth-init-fini-address-discrimination", i32 0}
