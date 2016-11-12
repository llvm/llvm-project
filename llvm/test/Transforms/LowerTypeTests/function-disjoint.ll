; RUN: opt -S -lowertypetests -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck --check-prefix=X64 %s
; RUN: opt -S -lowertypetests -mtriple=wasm32-unknown-unknown < %s | FileCheck --check-prefix=WASM32 %s

; Tests that we correctly handle bitsets with disjoint call target sets.

target datalayout = "e-p:64:64"

; X64:      module asm "f = .cfi.jumptable + 0"

; X64:      module asm ".cfi.jumptable:"
; X64-NEXT: module asm "jmp f.cfi@plt"
; X64-NEXT: module asm "int3"
; X64-NEXT: module asm "int3"
; X64-NEXT: module asm "int3"

; X64:      module asm "g = .cfi.jumptable.1 + 0"

; X64:      module asm ".cfi.jumptable.1:"
; X64-NEXT: module asm "jmp g.cfi@plt"
; X64-NEXT: module asm "int3"
; X64-NEXT: module asm "int3"
; X64-NEXT: module asm "int3"


; X64: @.cfi.jumptable = external hidden constant [1 x [8 x i8]]
; X64: @.cfi.jumptable.1 = external hidden constant [1 x [8 x i8]]

; WASM32: private constant [0 x i8] zeroinitializer
@0 = private unnamed_addr constant [2 x void ()*] [void ()* @f, void ()* @g], align 16

; X64: define internal void @f.cfi()
; WASM32: define void @f() !type !{{[0-9]+}} !wasm.index ![[I0:[0-9]+]]
define void @f() !type !0 {
  ret void
}

; X64: define internal void @g.cfi()
; WASM32: define void @g() !type !{{[0-9]+}} !wasm.index ![[I1:[0-9]+]]
define void @g() !type !1 {
  ret void
}

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  ; X64: icmp eq i64 {{.*}}, ptrtoint ([1 x [8 x i8]]* @.cfi.jumptable to i64)
  ; WASM32: icmp eq i64 {{.*}}, 1
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ; X64: icmp eq i64 {{.*}}, ptrtoint ([1 x [8 x i8]]* @.cfi.jumptable.1 to i64)
  ; WASM32: icmp eq i64 {{.*}}, 2
  %y = call i1 @llvm.type.test(i8* %p, metadata !"typeid2")
  %z = add i1 %x, %y
  ret i1 %z
}

; X64: declare void @f()
; X64: declare void @g()

; WASM32: ![[I0]] = !{i64 1}
; WASM32: ![[I1]] = !{i64 2}