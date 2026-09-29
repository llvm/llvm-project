; RUN: llvm-extract -func foo -S < %s | FileCheck %s
; RUN: llvm-extract -delete -func foo -S < %s | FileCheck --check-prefix=DELETE %s
; RUN: llvm-extract -alias zeda0 -S < %s | FileCheck --check-prefix=ALIAS %s
; RUN: llvm-extract -ralias '.*bar' -S < %s | FileCheck --check-prefix=ALIASRE %s

; Both aliases should be converted to declarations
; CHECK:      @zeda0 = external global i32
; CHECK:      define ptr @foo() {
; CHECK-NEXT:  call void @a0bar()
; CHECK-NEXT:  ret ptr @zeda0
; CHECK-NEXT: }
; CHECK:      declare void @a0bar()

; DELETE:      @zed = global i32 0
; DELETE:      @zeda0 = alias i32, ptr @zed
; DELETE-NEXT: @a0foo = alias ptr (), ptr @foo
; DELETE-NEXT: @a0a0bar = alias void (), ptr @bar
; DELETE-NEXT: @a0bar = alias void (), ptr @bar
; DELETE:      declare ptr @foo()
; DELETE:      define void @bar() {
; DELETE-NEXT:  %c = call ptr @foo()
; DELETE-NEXT:  ret void
; DELETE-NEXT: }

; ALIAS: @zed = external global i32
; ALIAS: @zeda0 = alias i32, ptr @zed

; ALIASRE: @a0a0bar = alias void (), ptr @bar
; ALIASRE: @a0bar = alias void (), ptr @bar
; ALIASRE: declare void @bar()

@zed = global i32 0
@zeda0 = alias i32, ptr @zed

@a0foo = alias ptr (), ptr @foo

define ptr @foo() {
  call void @a0bar()
  ret ptr @zeda0
}

@a0a0bar = alias void (), ptr @bar

@a0bar = alias void (), ptr @bar

define void @bar() {
  %c = call ptr @foo()
  ret void
}
