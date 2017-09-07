; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %S/Inputs/personality.ll -o %t2.bc

; RUN: llvm-lto2 run -o %t.o %t1.bc %t2.bc -save-temps \
; RUN:   -r %t2.bc,bar,p \
; RUN:   -r %t2.bc,personality_routine,p \
; RUN:   -r %t2.bc,personality_routine2,p \
; RUN:   -r %t2.bc,personality_routine3,p \
; RUN:   -r %t1.bc,foo,p \
; RUN:   -r %t1.bc,foo2,p \
; RUN:   -r %t1.bc,foo3,p \
; RUN:   -r %t1.bc,personality_routine,l \
; RUN:   -r %t1.bc,personality_routine2,l \
; RUN:   -r %t1.bc,personality_routine3,l \
; RUN:   -r %t1.bc,main,xp \
; RUN:   -r %t1.bc,bar,l
; RUN: llvm-readobj -t %t.o.1 | FileCheck %s --check-prefix=BINDING

; BINDING:     Symbol {
; BINDING:       Name: personality_routine
; BINDING-NEXT:  Value:
; BINDING-NEXT:  Size:
; BINDING-NEXT:  Binding: Global
; BINDING-NEXT:  Type: Function
; BINDING-NEXT:  Other [
; BINDING-NEXT:    STV_PROTECTED
; BINDING-NEXT:  ]
; BINDING-NEXT:  Section: .text
; BINDING-NEXT: }

; BINDING-NEXT: Symbol {
; BINDING-NEXT:  Name: personality_routine2
; BINDING-NEXT:  Value:
; BINDING-NEXT:  Size:
; BINDING-NEXT:  Binding: Global
; BINDING-NEXT:  Type: Function
; BINDING-NEXT:  Other [
; BINDING-NEXT:    STV_PROTECTED
; BINDING-NEXT:  ]
; BINDING-NEXT:  Section: .text
; BINDING-NEXT: }

; BINDING-NOT:  Name: personality_routine3

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare protected i32 @personality_routine(i32, i32, i64, i8*, i8*)
declare protected i32 @personality_routine2(i32, i32, i64, i8*, i8*)
declare protected i32 @personality_routine3(i32, i32, i64, i8*, i8*)
declare void @bar()

define void @foo() personality i32 (i32, i32, i64, i8*, i8*)* @personality_routine {
  ret void
}

define internal void @foo2b() personality i8* bitcast (i32 (i32, i32, i64, i8*, i8*)* @personality_routine2 to i8*) {
  ret void
}

define internal void @foo2a() prologue void ()* @foo2b {
  ret void
}

define void @foo2() prefix void ()* @foo2a {
  ret void
}

define void @foo3() personality i8* bitcast (i32 (i32, i32, i64, i8*, i8*)* @personality_routine3 to i8*) {
  ret void
}

define i32 @main() {
  call void @foo()
  call void @foo2()
  call void @bar()
  ret i32 0
}

