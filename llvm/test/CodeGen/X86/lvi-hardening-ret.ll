; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown < %s | FileCheck %s

define dso_local void @one_instruction() #0 {
; CHECK-LABEL: one_instruction:
entry:
  ret void
; CHECK-NOT:   retq
; CHECK:       popq %[[x:[^ ]*]]
; CHECK-NEXT:  lfence
; CHECK-NEXT:  jmpq *%[[x]]
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @ordinary_function(i32 %x, i32 %y) #0 {
; CHECK-LABEL: ordinary_function:
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %1 = load i32, i32* %y.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
; CHECK-NOT:  retq
; CHECK:      popq %[[x:[^ ]*]]
; CHECK-NEXT: lfence
; CHECK-NEXT: jmpq *%[[x]]
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @no_caller_saved_registers_function(i32 %x, i32 %y) #1 {
; CHECK-LABEL: no_caller_saved_registers_function:
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %1 = load i32, i32* %y.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
; CHECK-NOT:  retq
; CHECK:      shlq $0, (%{{[^ ]*}})
; CHECK-NEXT: lfence
; CHECK-NEXT: retq
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local preserve_mostcc void @preserve_most() #0 {
; CHECK-LABEL: preserve_most:
entry:
  ret void
; CHECK-NOT:  retq
; CHECK:      popq %r11
; CHECK-NEXT: lfence
; CHECK-NEXT: jmpq *%r11
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local preserve_allcc void @preserve_all() #0 {
; CHECK-LABEL: preserve_all:
entry:
  ret void
; CHECK-NOT:  retq
; CHECK:      popq %r11
; CHECK-NEXT: lfence
; CHECK-NEXT: jmpq *%r11
}

attributes #0 = { "target-features"="+lvi-cfi" }
attributes #1 = { "no_caller_saved_registers" "target-features"="+lvi-cfi" }
