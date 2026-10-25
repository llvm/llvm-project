; Global symbols that inline asm in the body of a function defines or refers to
; are symbols of the module, as those of module level asm are.

; RUN: opt -o %t %s
; RUN: llvm-lto2 dump-symtab %t | FileCheck %s --implicit-check-not=local_label --implicit-check-not=local_target --implicit-check-not=plain_name

; CHECK-DAG: D------X alternatives
; CHECK-DAG: D------X plain
; CHECK-DAG: D------X carrier
; CHECK-DAG: D------X from_att
; CHECK-DAG: D------X rtti_object
; CHECK-DAG: D--W---X weak_object
; CHECK-DAG: DU-----X referenced_object

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The operands are not known when the symbols are collected. $0 stands for an
; immediate here and $$ for a dollar sign.
; A reference to a value with local linkage is resolved inside the module.
@local_target = internal global i32 0
@global_target = global i32 0

define void @carrier() {
  call void asm sideeffect ".globl rtti_object\0Artti_object:\0A.long ${0:c}\0A.quad referenced_object\0A.quad local_target\0A.quad global_target\0Alocal_label:\0A.weak weak_object\0Aweak_object:\0A", "i"(i32 7)
  ret void
}

; Dialect alternatives pick the text of the dialect in use.
define void @alternatives() {
  call void asm sideeffect ".globl from_att\0Afrom_att:\0A$(movq $$1, %rax$|mov rax, 1$)\0A", ""()
  ret void
}

; Asm that defines no global symbol is not looked at.
define void @plain() {
  call void asm sideeffect "mfence\0A.long plain_name\0A", ""()
  ret void
}
