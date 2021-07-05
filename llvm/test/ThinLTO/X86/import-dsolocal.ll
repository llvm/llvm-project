; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.ll -o %t/a.bc
; RUN: opt -module-summary %t/b.ll -o %t/b.bc

;; With a small limit, *_aux are either imported declarations (external/linkonce_odr/weak_odr)
;; or unimported (linkonce/weak). Check we discard dso_local.
; RUN: llvm-lto2 run %t/a.bc %t/b.bc -o %t1 -save-temps -import-instr-limit=3 \
; RUN:   -r=%t/a.bc,main,plx -r=%t/a.bc,extern, -r=%t/a.bc,linkonce, -r=%t/a.bc,linkonceodr, -r=%t/a.bc,weak, -r=%t/a.bc,weakodr, \
; RUN:   -r=%t/b.bc,a,pl -r=%t/b.bc,b,pl -r=%t/b.bc,extern,pl -r=%t/b.bc,extern_aux,pl \
; RUN:   -r=%t/b.bc,linkonce,pl -r=%t/b.bc,linkonce_aux,pl -r=%t/b.bc,linkonceodr,pl -r=%t/b.bc,linkonceodr_aux,pl \
; RUN:   -r=%t/b.bc,weak,pl -r=%t/b.bc,weak_aux,pl -r=%t/b.bc,weakodr,pl -r=%t/b.bc,weakodr_aux,pl
; RUN: llvm-dis %t1.1.3.import.bc -o - | FileCheck %s --check-prefixes=DEST,DEST1

;; With a large limit, *_aux are either imported definitions (external/linkonce_odr/weak_odr)
;; or unimported (linkonce/weak). Check we discard dso_local as well.
; RUN: llvm-lto2 run %t/a.bc %t/b.bc -o %t2 -save-temps -import-instr-limit=10 \
; RUN:   -r=%t/a.bc,main,plx -r=%t/a.bc,extern, -r=%t/a.bc,linkonce, -r=%t/a.bc,linkonceodr, -r=%t/a.bc,weak, -r=%t/a.bc,weakodr, \
; RUN:   -r=%t/b.bc,a,pl -r=%t/b.bc,b,pl -r=%t/b.bc,extern,pl -r=%t/b.bc,extern_aux,pl \
; RUN:   -r=%t/b.bc,linkonce,pl -r=%t/b.bc,linkonce_aux,pl -r=%t/b.bc,linkonceodr,pl -r=%t/b.bc,linkonceodr_aux,pl \
; RUN:   -r=%t/b.bc,weak,pl -r=%t/b.bc,weak_aux,pl -r=%t/b.bc,weakodr,pl -r=%t/b.bc,weakodr_aux,pl
; RUN: llvm-dis %t2.1.3.import.bc -o - | FileCheck %s --check-prefixes=DEST,DEST2

; DEST:      @a = available_externally global i32 42, align 4
; DEST-NEXT: @b = external global i32*, align 8
; DEST:      declare void @linkonce()
; DEST:      declare void @weak()
; DEST:      define dso_local i32 @main()
; DEST:      define available_externally void @extern()

; DEST1:     declare i32 @extern_aux(i32*, i32**)
; DEST1:     declare i32 @linkonceodr_aux(i32*, i32**)
; DEST2:     define available_externally i32 @extern_aux(i32* %a, i32** %b)
; DEST2:     define available_externally i32 @linkonceodr_aux(i32* %a, i32** %b)

; DEST:      define available_externally void @weakodr()

; DEST1:     declare i32 @weakodr_aux(i32*, i32**)
; DEST2:     define available_externally i32 @weakodr_aux(i32* %a, i32** %b)

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @extern()
declare void @linkonce()
declare void @linkonceodr()
declare void @weak()
declare void @weakodr()

define i32 @main() {
  call void @extern()
  call void @linkonce()
  call void @linkonceodr()
  call void @weak()
  call void @weakodr()
  ret i32 0
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = dso_local global i32 42, align 4
@b = dso_local global i32* @a, align 8

define dso_local void @extern() {
  call i32 @extern_aux(i32* @a, i32** @b)
  ret void
}

define dso_local i32 @extern_aux(i32* %a, i32** %b) {
  %p = load i32*, i32** %b, align 8
  store i32 33, i32* %p, align 4
  %v = load i32, i32* %a, align 4
  ret i32 %v
}

define linkonce dso_local void @linkonce() {
  call i32 @linkonce_aux(i32* @a, i32** @b)
  ret void
}

define linkonce i32 @linkonce_aux(i32* %a, i32** %b) {
  %p = load i32*, i32** %b, align 8
  store i32 33, i32* %p, align 4
  %v = load i32, i32* %a, align 4
  ret i32 %v
}

define linkonce_odr dso_local void @linkonceodr() {
  call i32 @linkonceodr_aux(i32* @a, i32** @b)
  ret void
}

define linkonce_odr i32 @linkonceodr_aux(i32* %a, i32** %b) {
  %p = load i32*, i32** %b, align 8
  store i32 33, i32* %p, align 4
  %v = load i32, i32* %a, align 4
  ret i32 %v
}

define weak dso_local void @weak() {
  call i32 @weak_aux(i32* @a, i32** @b)
  ret void
}

define weak i32 @weak_aux(i32* %a, i32** %b) {
  %p = load i32*, i32** %b, align 8
  store i32 33, i32* %p, align 4
  %v = load i32, i32* %a, align 4
  ret i32 %v
}

define weak_odr dso_local void @weakodr() {
  call i32 @weakodr_aux(i32* @a, i32** @b)
  ret void
}

define weak_odr i32 @weakodr_aux(i32* %a, i32** %b) {
  %p = load i32*, i32** %b, align 8
  store i32 33, i32* %p, align 4
  %v = load i32, i32* %a, align 4
  ret i32 %v
}
