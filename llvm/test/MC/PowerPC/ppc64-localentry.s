
# RUN: llvm-mc -triple powerpc64-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -h -r --symbols | FileCheck %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -h -r --symbols | FileCheck %s

	.type callee1, @function
callee1:
	nop
	nop
	.localentry callee1, .-callee1
	nop
	nop
	.size callee1, .-callee1

	.type callee2, @function
callee2:
	nop
	nop
	.size callee2, .-callee2

	.type caller, @function
caller:
	bl callee1
	nop
	bl callee2
	nop
	.size caller, .-caller

	.section .text.other
caller_other:
	bl callee1
	nop
	bl callee2
	nop
	.size caller_other, .-caller_other

copy1 = callee1
copy2 = callee2

# Verify that use of .localentry implies ABI version 2
# CHECK: ElfHeader {
# CHECK: Flags [ (0x2)

# Verify that fixups to local function symbols are performed only
# if the target symbol does not use .localentry
# CHECK: Relocations [
# CHECK: Section ({{[0-9]*}}) .rela.text {
# CHECK-NEXT: R_PPC64_REL24 callee1
# CHECK-NEXT: }
# CHECK-NOT: R_PPC64_REL24 callee2
# CHECK: Section ({{[0-9]*}}) .rela.text.other {
# CHECK-NEXT: R_PPC64_REL24 callee1
# CHECK-NEXT: R_PPC64_REL24 .text
# CHECK-NEXT: }

# Verify that .localentry is encoded in the Other field.
# CHECK: Symbols [
# CHECK:       Name: callee1
# CHECK-NEXT:  Value:
# CHECK-NEXT:  Size: 16
# CHECK-NEXT:  Binding: Local
# CHECK-NEXT:  Type: Function
# CHECK-NEXT:  Other [ (0x60)
# CHECK-NEXT:  ]
# CHECK-NEXT:  Section: .text
# CHECK:       Name: callee2
# CHECK-NEXT:  Value:
# CHECK-NEXT:  Size: 8
# CHECK-NEXT:  Binding: Local
# CHECK-NEXT:  Type: Function
# CHECK-NEXT:  Other: 0
# CHECK-NEXT:  Section: .text

# Verify that symbol assignment copies the Other bits.
# CHECK:       Name: copy1
# CHECK-NEXT:  Value:
# CHECK-NEXT:  Size: 16
# CHECK-NEXT:  Binding: Local
# CHECK-NEXT:  Type: Function
# CHECK-NEXT:  Other [ (0x60)
# CHECK-NEXT:  ]
# CHECK-NEXT:  Section: .text
# CHECK:       Name: copy2
# CHECK-NEXT:  Value:
# CHECK-NEXT:  Size: 8
# CHECK-NEXT:  Binding: Local
# CHECK-NEXT:  Type: Function
# CHECK-NEXT:  Other: 0
# CHECK-NEXT:  Section: .text

