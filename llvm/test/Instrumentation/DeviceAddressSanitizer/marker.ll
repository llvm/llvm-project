; RUN: opt < %s -passes=dasan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "amdgcn-amd-amdhsa"

; CHECK: @__dasan_instrumented = linkonce_odr protected addrspace(1) constant i8 1

define internal i32 @checked(ptr addrspace(1) %p) sanitize_device_address {
  %v = load i32, ptr addrspace(1) %p, align 4
  ret i32 %v
}

; RUN: opt < %s -passes=dasan -o %t.a.bc
; RUN: opt < %s -passes=dasan -o %t.b.bc
; RUN: llvm-link %t.a.bc %t.b.bc -S | FileCheck %s --check-prefix=LINKED
; LINKED: @__dasan_instrumented = linkonce_odr protected addrspace(1) constant i8 1
; LINKED-NOT: @__dasan_instrumented
