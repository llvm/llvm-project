; RUN: llvm-as %s  -o - | llvm-dis - | FileCheck %s -check-prefixes CHECK,PROG-AS0
; RUN: llvm-as -data-layout "P200" %s  -o - | llvm-dis | FileCheck %s -check-prefixes CHECK,PROG-AS200
; RUN: not llvm-as -data-layout "P123456789" %s -o /dev/null 2>&1 | FileCheck %s -check-prefix BAD-DATALAYOUT
; BAD-DATALAYOUT: error: address space must be a 24-bit integer

; PROG-AS0-NOT: target datalayout
; PROG-AS200: target datalayout = "P200"

; Check that a function declaration without an address space (i.e. AS0) does not
; have the addrspace() attribute printed if it is address space zero and it is
; equal to the program address space.

; PROG-AS0: define void @no_as() {
; PROG-AS200: define void @no_as() addrspace(200) {
define void @no_as() {
  ret void
}

; A function with an explicit addrspace should only have the addrspace printed
; if it is non-zero or if the module has a nonzero datalayout
; PROG-AS0: define void @explit_as0()  {
; PROG-AS200: define void @explit_as0() addrspace(0) {
define void @explit_as0() addrspace(0) {
  ret void
}

; CHECK: define void @explit_as200() addrspace(200) {
define void @explit_as200() addrspace(200) {
  ret void
}

; CHECK: define void @explicit_as3() addrspace(3) {
define void @explicit_as3() addrspace(3) {
  ret void
}
