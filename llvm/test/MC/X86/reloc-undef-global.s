// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o - | llvm-readobj -r - | FileCheck --check-prefix=ELF %s
// RUN: llvm-mc -filetype=obj -triple x86_64-apple-darwin %s -o - | llvm-readobj -r - | FileCheck --check-prefix=MACHO %s


bar = foo + 4
	.globl bar
	.long bar

// ELF:      Relocations [
// ELF-NEXT:   Section {{.*}} .rela.text {
// ELF-NEXT:     0x0 R_X86_64_32 foo 0x4
// ELF-NEXT:   }
// ELF-NEXT: ]


// MACHO: Relocations [
// MACHO:   Section __text {
// MACHO:     0x0 0 2 1 X86_64_RELOC_UNSIGNED 0 bar
// MACHO:   }
// MACHO: ]
