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
define void @simple_align16(ptr addrspace(4) %ptr.out) align 32 {
entry:
  %out = load ptr addrspace(1), ptr addrspace(4) %ptr.out
  store i32 0, ptr addrspace(1) %out
  ret void
}
