;; Check that we can parse/print symbolic addres space for NVPTX triple. 
; RUN: split-file %s %t --leading-lines
; RUN: llvm-as < %t/num-to-sym.ll | llvm-dis | FileCheck %t/num-to-sym.ll
; RUN: llvm-as < %t/sym-to-sym.ll | llvm-dis | FileCheck %t/sym-to-sym.ll
; RUN: not llvm-as < %t/invalid-sym.ll 2>&1 | FileCheck %t/invalid-sym.ll

;--- num-to-sym.ll
target triple = "nvptx64-nvidia-cuda"

; CHECK: @str0 = private addrspace("global") constant
@str0 = private addrspace(1) constant [4 x i8] c"str\00"

; CHECK: @gvs = external addrspace("shared") global i32
@gvs = external addrspace(3) global i32

; CHECK: @gv_unknown_as = external addrspace(100) global i32
@gv_unknown_as = external addrspace(100) global i32

define void @foo() {
  ; CHECK: %alloca = alloca i32, align 4, addrspace("local")
  %alloca = alloca i32, addrspace(5)
  ; CHECK: store i32 0, ptr addrspace("local") %alloca
  store i32 0, ptr addrspace(5) %alloca
  ; CHECK: store i32 3, ptr addrspace("shared") @gvs
  store i32 3, ptr addrspace(3) @gvs
  ret void
}

;--- sym-to-sym.ll
target triple = "nvptx64-nvidia-cuda"

; CHECK: @str0 = private addrspace("global") constant
@str0 = private addrspace("global") constant [4 x i8] c"str\00"

; CHECK: @gvs = external addrspace("shared") global i32
@gvs = external addrspace("shared") global i32

; CHECK: @gv_unknown_as = external addrspace(100) global i32
@gv_unknown_as = external addrspace(100) global i32

define void @foo() {
  ; CHECK: %alloca = alloca i32, align 4, addrspace("local")
  %alloca = alloca i32, addrspace("local")
  ; CHECK: store i32 0, ptr addrspace("local") %alloca
  store i32 0, ptr addrspace("local") %alloca
  ; CHECK: store i32 3, ptr addrspace("shared") @gvs
  store i32 3, ptr addrspace("shared") @gvs
  ret void
}

;--- invalid-sym.ll
target triple = "nvptx64-nvidia-cuda"
; CHECK: error: invalid symbolic addrspace 'ram'
@str0 = private addrspace("ram") constant [4 x i8] c"str\00"
