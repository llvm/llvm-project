;; Check support for printing and parsing of address space names specified in
;; the datalayout.
; RUN: split-file %s %t --leading-lines
; RUN: llvm-as < %t/num-to-sym.ll | llvm-dis --print-addrspace-name=true  | FileCheck %t/num-to-sym.ll
; RUN: llvm-as < %t/sym-to-sym.ll | llvm-dis --print-addrspace-name=true  | FileCheck %t/sym-to-sym.ll
; RUN: llvm-as < %t/sym-to-num.ll | llvm-dis --print-addrspace-name=false | FileCheck %t/sym-to-num.ll
; RUN: not llvm-as < %t/invalid-name.ll 2>&1 | FileCheck %t/invalid-name.ll

;--- num-to-sym.ll
target datalayout = "P11-p2(global):32:8-p8(stack):8:8-p11(code):8:8"
; CHECK: target datalayout = "P11-p2(global):32:8-p8(stack):8:8-p11(code):8:8"

; CHECK: @str = private addrspace("global") constant [4 x i8] c"str\00"
@str = private addrspace(2) constant [4 x i8] c"str\00"

define void @foo() {
  ; CHECK: %alloca = alloca i32, align 4, addrspace("stack")
  %alloca = alloca i32, addrspace(8)
  ret void
}

; CHECK: define void @bar() addrspace("code")
define void @bar() addrspace(11) {
  ; CHECK: call addrspace("code") void @foo()
  call addrspace(11) void @foo()
  ret void
}

;--- sym-to-sym.ll
target datalayout = "P11-p2(global):32:8-p8(stack):8:8-p11(code):8:8"
; CHECK: target datalayout = "P11-p2(global):32:8-p8(stack):8:8-p11(code):8:8"

; CHECK: @str = private addrspace("global") constant [4 x i8] c"str\00"
@str = private addrspace("global") constant [4 x i8] c"str\00"

define void @foo() {
  ; CHECK: %alloca = alloca i32, align 4, addrspace("stack")
  %alloca = alloca i32, addrspace("stack")
  ret void
}

; CHECK: define void @bar() addrspace("code")
define void @bar() addrspace(11) {
  ; CHECK: call addrspace("code") void @foo()
  call addrspace("code") void @foo()
  ret void
}

;--- sym-to-num.ll
target datalayout = "P11-p2(global):32:8-p8(stack):8:8-p11(code):8:8"
; CHECK: target datalayout = "P11-p2(global):32:8-p8(stack):8:8-p11(code):8:8"

; CHECK: @str = private addrspace(2) constant [4 x i8] c"str\00"
@str = private addrspace("global") constant [4 x i8] c"str\00"

define void @foo() {
  ; CHECK: %alloca = alloca i32, align 4, addrspace(8)
  %alloca = alloca i32, addrspace("stack")
  ret void
}

; CHECK: define void @bar() addrspace(11)
define void @bar() addrspace(11) {
  ; CHECK: call addrspace(11) void @foo()
  call addrspace("code") void @foo()
  ret void
}

;--- invalid-name.ll
target datalayout = "P11-p2(global):32:8-p8(stack):8:8-p11(code):8:8"
; CHECK: error: invalid symbolic addrspace 'global3'
@str = private addrspace("global3") constant [4 x i8] c"str\00"

