; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri < %s | FileCheck -check-prefix=HSA %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -filetype=obj < %s | llvm-readobj --symbols -S --sd - | FileCheck -check-prefix=ELF %s

; ELF: Section {
; ELF: Name: .text
; ELF: SHF_ALLOC (0x2)
; ELF: SHF_EXECINSTR (0x4)
; ELF: AddressAlignment: 32
; ELF: }

; HSA: .globl simple_align16
; HSA: .p2align 5
define void @simple_align16(i32 addrspace(1)* addrspace(4)* %ptr.out) align 32 {
entry:
  %out = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %ptr.out
  store i32 0, i32 addrspace(1)* %out
  ret void
}
