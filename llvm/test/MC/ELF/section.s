// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s | FileCheck %s

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
.section "foo"

// CHECK:        Section {
// CHECK:          Name: bar-"foo"
// CHECK:        Section {
// CHECK:          Name: foo
