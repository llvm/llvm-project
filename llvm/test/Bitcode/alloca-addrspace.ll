; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target datalayout = "A2"

; CHECK-LABEL: define ptr addrspace(2) @alloca_addrspace_2() {
; CHECK: %alloca = alloca i8, align 1, addrspace(2)
define i8 addrspace(2)* @alloca_addrspace_2() {
  %alloca = alloca i8, addrspace(2)
  ret i8 addrspace(2)* %alloca
}

; CHECK-LABEL: define ptr addrspace(5) @alloca_addrspace_5() {
; CHECK: %alloca = alloca i8, align 1, addrspace(5)
define i8 addrspace(5)* @alloca_addrspace_5() {
  %alloca = alloca i8, addrspace(5)
  ret i8 addrspace(5)* %alloca
}

