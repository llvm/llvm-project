# RUN: rm -rf %t && split-file %s %t
; REQUIRES: default_target
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
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @constructor, i8* null }]
declare void @constructor()

;--- hasDtor.ll
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @destructor, i8* null }]
declare void @destructor()

;--- hasBoth.ll
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @constructor, i8* null }]
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @destructor, i8* null }]
declare void @constructor()
declare void @destructor()

;--- hasNone.ll
declare void @foo()


