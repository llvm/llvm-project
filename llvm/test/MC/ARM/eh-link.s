@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S  | FileCheck %s

@ Test that the ARM_EXIDX sections point (Link) to the corresponding text
@ sections.

@ FIXME: The section numbers are not important. If llvm-readobj printed the
@ name first we could use a FileCheck variable.

@ CHECK:      Section {
@ CHECK:        Index: 4
@ CHECK-NEXT:   Name: .text
@ CHECK-NEXT:   Type: SHT_PROGBITS
@ CHECK-NEXT:   Flags [
@ CHECK-NEXT:     SHF_ALLOC
@ CHECK-NEXT:     SHF_EXECINSTR
@ CHECK-NEXT:     SHF_GROUP
@ CHECK-NEXT:   ]
@ CHECK-NEXT:   Address: 0x0
@ CHECK-NEXT:   Offset:
@ CHECK-NEXT:   Size: 4
@ CHECK-NEXT:   Link: 0
@ CHECK-NEXT:   Info: 0
@ CHECK-NEXT:   AddressAlignment: 1
@ CHECK-NEXT:   EntrySize: 0
@ CHECK-NEXT: }
@ CHECK-NEXT: Section {
@ CHECK-NEXT:   Index: 5
@ CHECK-NEXT:   Name: .ARM.exidx
@ CHECK-NEXT:   Type: SHT_ARM_EXIDX
@ CHECK-NEXT:   Flags [
@ CHECK-NEXT:     SHF_ALLOC
@ CHECK-NEXT:     SHF_GROUP
@ CHECK-NEXT:     SHF_LINK_ORDER
@ CHECK-NEXT:   ]
@ CHECK-NEXT:   Address: 0x0
@ CHECK-NEXT:   Offset:
@ CHECK-NEXT:   Size: 8
@ CHECK-NEXT:   Link: 4
@ CHECK-NEXT:   Info: 0
@ CHECK-NEXT:   AddressAlignment: 4
@ CHECK-NEXT:   EntrySize: 0
@ CHECK-NEXT: }

@ CHECK:      Section {
@ CHECK:        Index: 8
@ CHECK-NEXT:   Name: .text
@ CHECK-NEXT:   Type: SHT_PROGBITS
@ CHECK-NEXT:   Flags [
@ CHECK-NEXT:     SHF_ALLOC
@ CHECK-NEXT:     SHF_EXECINSTR
@ CHECK-NEXT:     SHF_GROUP
@ CHECK-NEXT:   ]
@ CHECK-NEXT:   Address: 0x0
@ CHECK-NEXT:   Offset:
@ CHECK-NEXT:   Size: 4
@ CHECK-NEXT:   Link: 0
@ CHECK-NEXT:   Info: 0
@ CHECK-NEXT:   AddressAlignment: 1
@ CHECK-NEXT:   EntrySize: 0
@ CHECK-NEXT: }
@ CHECK-NEXT: Section {
@ CHECK-NEXT:   Index: 9
@ CHECK-NEXT:   Name: .ARM.exidx
@ CHECK-NEXT:   Type: SHT_ARM_EXIDX
@ CHECK-NEXT:   Flags [
@ CHECK-NEXT:     SHF_ALLOC
@ CHECK-NEXT:     SHF_GROUP
@ CHECK-NEXT:     SHF_LINK_ORDER
@ CHECK-NEXT:   ]
@ CHECK-NEXT:   Address: 0x0
@ CHECK-NEXT:   Offset:
@ CHECK-NEXT:   Size: 8
@ CHECK-NEXT:   Link: 8
@ CHECK-NEXT:   Info: 0
@ CHECK-NEXT:   AddressAlignment: 4
@ CHECK-NEXT:   EntrySize: 0
@ CHECK-NEXT: }

	.section	.text,"axG",%progbits,f,comdat
f:
	.fnstart
	mov	pc, lr
	.fnend

	.section	.text,"axG",%progbits,g,comdat
g:
	.fnstart
	mov	pc, lr
	.fnend
