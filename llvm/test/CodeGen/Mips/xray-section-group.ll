; RUN: llc -filetype=asm -o - -mtriple=mips-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -filetype=asm -o - -mtriple=mipsel-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -filetype=obj -o %t -mtriple=mips-unknown-linux-gnu -function-sections < %s
; RUN: llvm-readobj -sections %t | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llc -filetype=obj -o %t -mtriple=mipsel-unknown-linux-gnu -function-sections < %s
; RUN: llvm-readobj -sections %t | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llc -filetype=asm -o - -mtriple=mips64-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -filetype=asm -o - -mtriple=mips64el-unknown-linux-gnu -function-sections < %s | FileCheck %s
; RUN: llc -filetype=obj -o %t -mtriple=mips64-unknown-linux-gnu -function-sections < %s
; RUN: llvm-readobj -sections %t | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: llc -filetype=obj -o %t -mtriple=mips64el-unknown-linux-gnu -function-sections < %s
; RUN: llvm-readobj -sections %t | FileCheck %s --check-prefix=CHECK-OBJ

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK: .section .text.foo,"ax",@progbits
  ret i32 0
; CHECK: .section xray_instr_map,"awo",@progbits,foo,unique,1
}

; CHECK-OBJ: Section {
; CHECK-OBJ:   Name: xray_instr_map

$bar = comdat any
define i32 @bar() nounwind noinline uwtable "function-instrument"="xray-always" comdat($bar) {
; CHECK: .section .text.bar,"axG",@progbits,bar,comdat
  ret i32 1
; CHECK: .section xray_instr_map,"aGwo",@progbits,bar,comdat,bar,unique,2
}

; CHECK-OBJ: Section {
; CHECK-OBJ:   Name: xray_instr_map
