# REQUIRES: asserts

# RUN: llvm-mc -triple=riscv32-linux-gnu -mattr=+relax -filetype=obj -o %t.32.o %s
# RUN: llvm-jitlink -noexec -phony-externals -debug-only=jitlink %t.32.o 2>&1 | \
# RUN:   FileCheck %s

# RUN: llvm-mc -triple=riscv64-linux-gnu -mattr=+relax -filetype=obj -o %t.64.o %s
# RUN: llvm-jitlink -noexec -phony-externals -debug-only=jitlink %t.64.o 2>&1 | \
# RUN:   FileCheck %s

# Check that splitting of eh-frame sections works.
#
# CHECK: DWARFRecordSectionSplitter: Processing .eh_frame...
# CHECK:  Processing block at
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = .eh_frame
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = .eh_frame
# CHECK: EHFrameEdgeFixer: Processing .eh_frame in "{{.*}}"...
# CHECK:   Processing block at
# CHECK:     Record is CIE
# CHECK:   Processing block at
# CHECK:     Record is FDE
# CHECK:       Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK:       Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK:       Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}
# CHECK:   Processing block at
# CHECK:     Record is FDE
# CHECK:       Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK:       Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK:       Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}

## This is "int main { throw 1; }" compiled for riscv32. We use the 32-bit
## version because it is also legal for riscv64.
	.text
	.globl	main
	.p2align	1
	.type	main,@function
main:
	.cfi_startproc
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sw	ra, 12(sp)
	.cfi_offset ra, -4
	li	a0, 4
	call	__cxa_allocate_exception
	li	a1, 1
	sw	a1, 0(a0)
	lga a1, _ZTIi
	li	a2, 0
	call	__cxa_throw
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc

	.globl	dup
	.p2align	1
	.type	dup,@function
dup:
	.cfi_startproc
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sw	ra, 12(sp)
	.cfi_offset ra, -4
	li	a0, 4
	call	__cxa_allocate_exception
	li	a1, 1
	sw	a1, 0(a0)
	lga a1, _ZTIi
	li	a2, 0
	call	__cxa_throw
.Lfunc_end1:
	.size	dup, .Lfunc_end1-dup
	.cfi_endproc
