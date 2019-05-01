// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S | FileCheck  %s

// Test that mergeable constants have sh_entsize set.

// 1 byte strings
    .section	.rodata.str1.1,"aMS",@progbits,1

    .type	.L.str1,@object         # @.str1
.L.str1:
	.asciz	 "tring"
	.size	.L.str1, 6

	.type	.L.str2,@object         # @.str2
.L.str2:
	.asciz	 "String"
	.size	.L.str2, 7

// 2 byte strings
    .section	.rodata.str2.1,"aMS",@progbits,2
	.type	.L.str3,@object         # @.str3
.L.str3:
	.asciz	 "L\000o\000n\000g\000"
	.size	.L.str3, 9

	.type	.L.str4,@object         # @.str4
.L.str4:
	.asciz	 "o\000n\000g\000"
	.size	.L.str4, 7

 // 8 byte constants
    .section	.rodata.cst8,"aM",@progbits,8
    .quad 42
    .quad 42

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .rodata.str1.1
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_MERGE
// CHECK-NEXT:       SHF_STRINGS
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 13
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 1
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .rodata.str2.1
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_MERGE
// CHECK-NEXT:       SHF_STRINGS
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 16
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 2
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .rodata.cst8
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_MERGE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 16
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 8
// CHECK-NEXT:   }
