; When the target machine has no feature string of its own, the module's ABI
; flags are derived from a function's target attributes. Check the two cases
; where the function they used to be read from carries no attributes: a module
; starting with a declaration, and a module with no functions at all.

; RUN: split-file %s %t
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32 \
; RUN:     %t/single-float.ll -o %t/single-float.o
; RUN: llvm-readobj -A %t/single-float.o | FileCheck %s -check-prefix=SINGLE
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32 \
; RUN:     %t/soft-float.ll -o %t/soft-float.o
; RUN: llvm-readobj -A %t/soft-float.o | FileCheck %s -check-prefix=SOFT
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 \
; RUN:     %t/soft-float.ll -o - | FileCheck %s -check-prefix=SOFT-ASM
; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux -mcpu=mips32 \
; RUN:     %t/no-functions.ll -o %t/no-functions.o
; RUN: llvm-readobj -A %t/no-functions.o | FileCheck %s -check-prefix=NONE

;--- single-float.ll
; SINGLE: FP ABI: Hard float (single precision)

declare void @a_declaration()

define dso_local void @a_definition() #0 {
  call void @a_declaration()
  ret void
}

attributes #0 = { "target-features"="+single-float" }

;--- soft-float.ll
; SOFT: FP ABI: Soft float
; SOFT-ASM: .module softfloat

declare void @another_declaration()

define dso_local void @another_definition() "use-soft-float"="true" {
  call void @another_declaration()
  ret void
}

;--- no-functions.ll
; A module with no functions says nothing about the FP ABI it was built for,
; so it must not claim one. LTO produces such modules.
; NONE: FP ABI: Hard or soft float

@a_global = global i32 0
