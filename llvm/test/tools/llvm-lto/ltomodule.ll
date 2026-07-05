# RUN: rm -rf %t && split-file %s %t
; REQUIRES: default_triple
; RUN: llvm-as < %t/hasCtor.ll > %t.bc
; RUN: llvm-lto %t.bc -query-hasCtorDtor | FileCheck %s --check-prefixes=POSITIVE

; RUN: llvm-as < %t/hasDtor.ll > %t.bc
; RUN: llvm-lto %t.bc -query-hasCtorDtor | FileCheck %s --check-prefixes=POSITIVE

; RUN: llvm-as < %t/hasBoth.ll > %t.bc
; RUN: llvm-lto %t.bc -query-hasCtorDtor | FileCheck %s --check-prefixes=POSITIVE

; RUN: llvm-as < %t/hasNone.ll > %t.bc
; RUN: llvm-lto %t.bc -query-hasCtorDtor | FileCheck %s --check-prefixes=NEGATIVE

; POSITIVE: .bc: hasCtorDtor = true
; NEGATIVE: .bc: hasCtorDtor = false

;--- hasCtor.ll
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @constructor, ptr null }]
declare void @constructor()

;--- hasDtor.ll
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @destructor, ptr null }]
declare void @destructor()

;--- hasBoth.ll
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @constructor, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @destructor, ptr null }]
declare void @constructor()
declare void @destructor()

;--- hasNone.ll
declare void @foo()


