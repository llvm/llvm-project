; RUN: llvm-extract -func foo -S < %s | FileCheck %s
; RUN: llvm-extract -delete -func foo -S < %s | FileCheck --check-prefix=DELETE %s
; RUN: llvm-extract -alias zeda0 -S < %s | FileCheck --check-prefix=ALIAS %s
; RUN: llvm-extract -ralias .*bar -S < %s | FileCheck --check-prefix=ALIASRE %s

; Both aliases should be converted to declarations
; CHECK:      @zeda0 = external global i32
; CHECK:      define i32* @foo() {
; CHECK-NEXT:  call void @a0bar()
; CHECK-NEXT:  ret i32* @zeda0
; CHECK-NEXT: }
; CHECK:      declare void @a0bar()

; DELETE:      @zed = global i32 0
; DELETE:      @zeda0 = alias i32* @zed
; DELETE-NEXT: @a0foo = alias i32* ()* @foo
; DELETE-NEXT: @a0a0bar = alias void ()* @a0bar
; DELETE-NEXT: @a0bar = alias void ()* @bar
; DELETE:      declare i32* @foo()
; DELETE:      define void @bar() {
; DELETE-NEXT:  %c = call i32* @foo()
; DELETE-NEXT:  ret void
; DELETE-NEXT: }

; ALIAS: @zed = external global i32
; ALIAS: @zeda0 = alias i32* @zed

; ALIASRE: @a0a0bar = alias void ()* @a0bar
; ALIASRE: @a0bar = alias void ()* @bar
; ALIASRE: declare void @bar()

@zed = global i32 0
@zeda0 = alias i32* @zed

@a0foo = alias i32* ()* @foo

define i32* @foo() {
  call void @a0bar()
  ret i32* @zeda0
}

@a0a0bar = alias void ()* @a0bar

@a0bar = alias void ()* @bar

define void @bar() {
  %c = call i32* @foo()
  ret void
}
