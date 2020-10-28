# RUN: llvm-mc -triple mips-unknown-linux -filetype=obj \
# RUN:     -mips-round-section-sizes %s | llvm-readobj --sections - | FileCheck %s
	.section ".talign1", "ax"
	.p2align 4
t1:	.byte 1

	.section ".talign2", "ax"
	.p2align 3
t2:	addiu $2, $2, 1
	addiu $2, $2, 1

	.section ".talign3", "ax"
	.p2align 3
t3:	addiu $2, $2, 1

	.section ".talign4", "ax"
t4:	.byte 1

	.section ".dalign1", "a"
	.p2align 4
d1:	.byte 1

	.section ".dalign2", "a"
	.p2align 3
d2:	.word 1
        .word 2

	.section ".dalign3", "a"
	.p2align 3
d3:	.word 1

	.section ".dalign4", "a"
d4:	.byte 1

	.section ".dalign5", "a"
	.p2align 16
d5:	.word 1

	.section ".nalign1", ""
	.p2align 4
n1:	.byte 1

	.section ".nalign2", ""
	.p2align 3
n2:	.word 1
        .word 2

	.section ".nalign3", ""
	.p2align 3
n3:	.word 1

	.section ".nalign4", ""
n4:	.byte 1

# CHECK-LABEL:   Name: .talign1
# CHECK:         Size: 16
# CHECK:         AddressAlignment: 16
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .talign2
# CHECK:         Size: 8
# CHECK:         AddressAlignment: 8
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .talign3
# CHECK:         Size: 8
# CHECK:         AddressAlignment: 8
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .talign4
# CHECK:         Size: 1
# CHECK:         AddressAlignment: 1
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .dalign1
# CHECK:         Size: 16
# CHECK:         AddressAlignment: 16
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .dalign2
# CHECK:         Size: 8
# CHECK:         AddressAlignment: 8
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .dalign3
# CHECK:         Size: 8
# CHECK:         AddressAlignment: 8
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .dalign4
# CHECK:         Size: 1
# CHECK:         AddressAlignment: 1
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .dalign5
# CHECK:         Size: 65536
# CHECK:         AddressAlignment: 65536
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .nalign1
# CHECK:         Size: 16
# CHECK:         AddressAlignment: 16
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .nalign2
# CHECK:         Size: 8
# CHECK:         AddressAlignment: 8
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .nalign3
# CHECK:         Size: 8
# CHECK:         AddressAlignment: 8
# CHECK-LABEL: }
# CHECK-LABEL:   Name: .nalign4
# CHECK:         Size: 1
# CHECK:         AddressAlignment: 1
# CHECK-LABEL: }
