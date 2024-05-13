; RUN: llc -mtriple=mips-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -filetype=obj -o %t -mtriple=mips-unknown-linux-gnu -function-sections < %s
; RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llc -filetype=obj -o %t -mtriple=mipsel-unknown-linux-gnu -function-sections < %s
; RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llc -mtriple=mips64-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -mtriple=mips64el-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -filetype=obj -o %t -mtriple=mips64-unknown-linux-gnu -function-sections < %s
; RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llc -filetype=obj -o %t -mtriple=mips64el-unknown-linux-gnu -function-sections < %s
; RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix=CHECK-OBJ

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK: .section .text.foo,"ax",@progbits
  ret i32 0
; CHECK: .section xray_instr_map,"ao",@progbits,foo{{$}}
}

; CHECK-OBJ: Section {
; CHECK-OBJ:   Name: xray_instr_map

$bar = comdat any
define i32 @bar() nounwind noinline uwtable "function-instrument"="xray-always" comdat($bar) {
; CHECK: .section .text.bar,"axG",@progbits,bar,comdat
  ret i32 1
; CHECK: .section xray_instr_map,"aoG",@progbits,bar,bar,comdat{{$}}
}

; CHECK-OBJ: Section {
; CHECK-OBJ:   Name: xray_instr_map
