; RUN: llc < %s -mtriple=mips -mcpu=mips1    | FileCheck %s -check-prefixes=ALL,MIPS1
; RUN: llc < %s -mtriple=mips -mcpu=mips2    | FileCheck %s -check-prefixes=ALL,MIPS2
; RUN: llc < %s -mtriple=mips -mcpu=mips32r2 | FileCheck %s -check-prefixes=ALL,MIPS32
target datalayout = "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mipsel-unknown-unknown-elf"

; Function Attrs: noinline nounwind optnone
define dso_local i32 @add_two_pointers(i32* %a, i32* %b) #0 {
entry:
; ALL-LABEL: add_two_pointers:
  %a.addr = alloca i32*, align 4
  %b.addr = alloca i32*, align 4
  store i32* %a, i32** %a.addr, align 4
  store i32* %b, i32** %b.addr, align 4
  %0 = load i32*, i32** %a.addr, align 4
  %1 = load i32, i32* %0, align 4
  ; ALL:        lw $1, 4($fp)
  ; MIPS1:      nop
  ; MIPS2-NOT:  nop
  ; MIPS32-NOT: nop
  ; ALL:        lw $1, 0($1)
  %2 = load i32*, i32** %b.addr, align 4
  %3 = load i32, i32* %2, align 4
  ; ALL:        lw $2, 0($fp)
  ; MIPS1:      nop
  ; MIPS2-NOT:  nop
  ; MIPS32-NOT: nop
  ; ALL:        lw $2, 0($2)
  %add = add nsw i32 %1, %3
  ret i32 %add
  ; ALL:        lw $ra, 12($sp)
  ; MIPS1:      nop
  ; MIPS2-NOT:  nop
  ; MIPS32-NOT: nop
  ; ALL:        jr $ra
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="-noabicalls" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}

