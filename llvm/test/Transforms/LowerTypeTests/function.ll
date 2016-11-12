; RUN: opt -S -lowertypetests -mtriple=i686-unknown-linux-gnu < %s | FileCheck --check-prefix=X86 %s
; RUN: opt -S -lowertypetests -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck --check-prefix=X86 %s
; RUN: opt -S -lowertypetests -mtriple=arm-unknown-linux-gnu < %s | FileCheck --check-prefix=ARM %s
; RUN: opt -S -lowertypetests -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck --check-prefix=ARM %s
; RUN: opt -S -lowertypetests -mtriple=wasm32-unknown-unknown < %s | FileCheck --check-prefix=WASM32 %s

; Tests that we correctly handle bitsets containing 2 or more functions.

target datalayout = "e-p:64:64"

; X86:      module asm ".globl f"
; X86-NEXT: module asm ".type f, function"
; X86-NEXT: module asm "f = .cfi.jumptable + 0"
; X86-NEXT: module asm ".size f, 8"
; X86-NEXT: module asm ".type g, function"
; X86-NEXT: module asm "g = .cfi.jumptable + 8"
; X86-NEXT: module asm ".size g, 8"
; X86-NEXT: module asm ".section .text.cfi, \22ax\22, @progbits"
; X86-NEXT: module asm ".balign 8"
; X86-NEXT: module asm ".cfi.jumptable:"
; X86-NEXT: module asm "jmp f.cfi@plt"
; X86-NEXT: module asm "int3"
; X86-NEXT: module asm "int3"
; X86-NEXT: module asm "int3"
; X86-NEXT: module asm "jmp g.cfi@plt"
; X86-NEXT: module asm "int3"
; X86-NEXT: module asm "int3"
; X86-NEXT: module asm "int3"

; ARM:      module asm ".globl f"
; ARM-NEXT: module asm ".type f, function"
; ARM-NEXT: module asm "f = .cfi.jumptable + 0"
; ARM-NEXT: module asm ".size f, 4"
; ARM-NEXT: module asm ".type g, function"
; ARM-NEXT: module asm "g = .cfi.jumptable + 4"
; ARM-NEXT: module asm ".size g, 4"
; ARM-NEXT: module asm ".section .text.cfi, \22ax\22, @progbits"
; ARM-NEXT: module asm ".balign 4"
; ARM-NEXT: module asm ".cfi.jumptable:"
; ARM-NEXT: module asm "b f.cfi"
; ARM-NEXT: module asm "b g.cfi"

; X86: @.cfi.jumptable = external hidden constant [2 x [8 x i8]]
; ARM: @.cfi.jumptable = external hidden constant [2 x [4 x i8]]

; WASM32: private constant [0 x i8] zeroinitializer
@0 = private unnamed_addr constant [2 x void (...)*] [void (...)* bitcast (void ()* @f to void (...)*), void (...)* bitcast (void ()* @g to void (...)*)], align 16

; X86: @llvm.used = appending global [2 x i8*] [i8* bitcast (void ()* @f.cfi to i8*), i8* bitcast (void ()* @g.cfi to i8*)], section "llvm.metadata"
; ARM: @llvm.used = appending global [2 x i8*] [i8* bitcast (void ()* @f.cfi to i8*), i8* bitcast (void ()* @g.cfi to i8*)], section "llvm.metadata"

; X86: define internal void @f.cfi()
; ARM: define internal void @f.cfi()
; WASM32: define void @f() !type !{{[0-9]+}} !wasm.index ![[I0:[0-9]+]]
define void @f() !type !0 {
  ret void
}

; X86: define internal void @g.cfi()
; ARM: define internal void @g.cfi()
; WASM32: define internal void @g() !type !{{[0-9]+}} !wasm.index ![[I1:[0-9]+]]
define internal void @g() !type !0 {
  ret void
}

!0 = !{i32 0, !"typeid1"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  ; X86: sub i64 {{.*}}, ptrtoint ([2 x [8 x i8]]* @.cfi.jumptable to i64)
  ; ARM: sub i64 {{.*}}, ptrtoint ([2 x [4 x i8]]* @.cfi.jumptable to i64)
  ; WASM32: sub i64 {{.*}}, 1
  ; WASM32: icmp ult i64 {{.*}}, 2
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}

; X86: declare void @f()
; ARM: declare void @f()
; X86: declare hidden void @g()
; ARM: declare hidden void @g()

; WASM32: ![[I0]] = !{i64 1}
; WASM32: ![[I1]] = !{i64 2}
