// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S | FileCheck %s
// RUN: llvm-mc -filetype=asm -triple x86_64-pc-linux-gnu %s -o - |  FileCheck %s --check-prefix=ASM

// Test that these names are accepted.

.section	.note.GNU-stack,"",@progbits
.section	.note.GNU-stack2,"",%progbits
.section	.note.GNU-,"",@progbits
.section	-.note.GNU,"","progbits"
.section	src/stack.c,"",@progbits
.section	~!@$%^&*()_-+={[}]|\\:<>,"",@progbits

// CHECK: Name: .note.GNU-stack
// CHECK: Name: .note.GNU-stack2
// CHECK: Name: .note.GNU-
// CHECK: Name: -.note.GNU
// CHECK: Name: src/stack.c
// CHECK: Name: ~!@$%^&*()_-+={[}]|\\:<>

// Test that the defaults are used

.section	.init
.section	.fini
.section	.rodata
.section	zed, ""

// CHECK:        Section {
// CHECK:          Name: .init
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .fini
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .rodata
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: zed
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }

.section	.note.test,"",@note
// CHECK:        Section {
// CHECK:          Name: .note.test
// CHECK-NEXT:     Type: SHT_NOTE
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }

// Test that we can parse these
foo:
bar:
.section        .text.foo,"axG",@progbits,foo,comdat
.section        .text.bar,"axMG",@progbits,42,bar,comdat

// Test that the default values are not used

.section .eh_frame,"a",@unwind

// CHECK:        Section {
// CHECK:          Name: .eh_frame
// CHECK-NEXT:     Type: SHT_X86_64_UNWIND
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }

.section .excluded,"e",@progbits

// CHECK:      Section {
// CHECK:        Name: .excluded
// CHECK-NEXT:   Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x80000000)
// CHECK-NEXT:     SHF_EXCLUDE (0x80000000)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: 0x0
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Link: 0
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 1
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT: }

// Test that we handle the strings like gas
.section bar-"foo"
.section "fooo"


// CHECK:        Section {
// CHECK:          Name: bar-"foo"
// CHECK:        Section {
// CHECK:          Name: fooo

// Test SHF_LINK_ORDER

.section .shf_metadata_target1, "a"
        .quad 0
.section .shf_metadata_target2, "a", @progbits, unique, 1
.Lshf_metadata_target2_1:
        .quad 0
.section .shf_metadata_target2, "a", @progbits, unique, 2
.Lshf_metadata_target2_2:
        .quad 0

.section .shf_metadata1,"ao",@progbits,.Lshf_metadata_target2_1
.section .shf_metadata2,"ao",@progbits,.Lshf_metadata_target2_2
.section .shf_metadata3,"ao",@progbits,.shf_metadata_target1
// ASM: .section .shf_metadata1,"ao",@progbits,.Lshf_metadata_target2_1
// ASM: .section .shf_metadata2,"ao",@progbits,.Lshf_metadata_target2_2
// ASM: .section .shf_metadata3,"ao",@progbits,.shf_metadata_target1

// CHECK:      Section {
// CHECK:        Index: 22
// CHECK-NEXT:   Name: .shf_metadata_target1

// CHECK:      Section {
// CHECK:        Index: 23
// CHECK-NEXT:   Name: .shf_metadata_target2

// CHECK:      Section {
// CHECK:        Index: 24
// CHECK-NEXT:   Name: .shf_metadata_target2

// CHECK:      Section {
// CHECK:        Name: .shf_metadata1
// CHECK-NEXT:   Type: SHT_PROGBITS
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:     SHF_LINK_ORDER
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address:
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Link:    23
// CHECK-NEXT:   Info:    0

// CHECK:      Section {
// CHECK:        Name: .shf_metadata2
// CHECK-NEXT:   Type: SHT_PROGBITS
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:     SHF_LINK_ORDER
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address:
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Link:    24
// CHECK-NEXT:   Info:    0

// CHECK:      Section {
// CHECK:        Name: .shf_metadata3
// CHECK-NEXT:   Type: SHT_PROGBITS
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_ALLOC
// CHECK-NEXT:     SHF_LINK_ORDER
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address:
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Link:    22
// CHECK-NEXT:   Info:    0

.section	.text.foo
// CHECK:        Section {
// CHECK:          Name: .text.foo
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]

.section .bss
// CHECK:        Section {
// CHECK:          Name: .bss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]

.section .bss.foo
// CHECK:        Section {
// CHECK:          Name: .bss.foo
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]

.section .tbss
// CHECK:        Section {
// CHECK:          Name: .tbss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]

.section .tbss.foo
// CHECK:        Section {
// CHECK:          Name: .tbss.foo
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_TLS
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]

// Test SHT_LLVM_ODRTAB

.section .odrtab,"e",@llvm_odrtab
// ASM: .section .odrtab,"e",@llvm_odrtab

// CHECK:        Section {
// CHECK:          Name: .odrtab
// CHECK-NEXT:     Type: SHT_LLVM_ODRTAB
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_EXCLUDE
// CHECK-NEXT:     ]

// Test SHT_LLVM_LINKER_OPTIONS

.section ".linker-options","e",@llvm_linker_options
// ASM: .section ".linker-options","e",@llvm_linker_options

// CHECK: Section {
// CHECK:   Name: .linker-options
// CHECK-NEXT:   Type: SHT_LLVM_LINKER_OPTIONS
// CHECK-NEXT:   Flags [
// CHECK-NEXT:     SHF_EXCLUDE
// CHECK-NEXT:   ]
// CHECK: }

