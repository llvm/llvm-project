# RUN: llvm-mc -triple=mipsel-unknown-linux < %s | FileCheck %s
# RUN: llvm-mc -triple=mipsel-unknown-linux < %s | \
# RUN:     llvm-mc -triple=mipsel-unknown-linux | FileCheck %s

        .text
branch:
	bnez $2, foo

# CHECK-LABEL: branch:
# CHECK:           bnez $2, foo
# CHECK:           nop
# CHECK-NOT:       nop

cprestore:
	.option pic2
	.cprestore 16
	jal foo

# CHECK-LABEL: cprestore:
# CHECK:           .cprestore 16
# CHECK:           lw $25, %call16(foo)($gp)
# CHECK:           jalr $25
# CHECK:           nop
# CHECK:           lw $gp, 16($sp)
# CHECK-NOT:       nop
# CHECK-NOT:       lw
