# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-nacl %s \
# RUN:   | llvm-objdump --no-print-imm-hex -d -z --no-show-raw-insn - | FileCheck %s

# This test tests that address-masking sandboxing is added when given assembly
# input.


# Test that address-masking sandboxing is added before indirect branches and
# returns.

	.align	4
test1:
	.set	noreorder

        jr	$a0
        nop
        jr	$ra
        nop

# CHECK-LABEL:   <test1>:

# CHECK:         and     $4, $4, $14
# CHECK-NEXT:    jr      $4

# Check that additional nop is inserted, to align mask and jr to the next
# bundle.

# CHECK-NEXT:    nop
# CHECK-NEXT:    nop

# CHECK:         and     $ra, $ra, $14
# CHECK-NEXT:    jr      $ra



# Test that address-masking sandboxing is added before load instructions.

	.align	4
test2:
	.set	noreorder

        lb      $4, 0($1)
        nop
        lbu     $4, 0($2)
        lh      $4, 0($3)
        lhu     $1, 0($4)
        lw      $4, 0($5)
        lwc1    $f0, 0($6)
        ldc1    $f2, 0($7)
        ll      $4, 0($8)
        lwl     $4, 0($9)
        lwr     $4, 0($10)

        lw      $4, 0($sp)
        lw      $4, 0($t8)

# CHECK-LABEL:   <test2>:

# CHECK:         and     $1, $1, $15
# CHECK-NEXT:    lb      $4, 0($1)

# Check that additional nop is inserted, to align mask and load to the next
# bundle.

# CHECK:         nop
# CHECK:         nop

# CHECK:         and     $2, $2, $15
# CHECK-NEXT:    lbu     $4, 0($2)

# CHECK:         and     $3, $3, $15
# CHECK-NEXT:    lh      $4, 0($3)

# CHECK:         and     $4, $4, $15
# CHECK-NEXT:    lhu     $1, 0($4)

# CHECK:         and     $5, $5, $15
# CHECK-NEXT:    lw      $4, 0($5)

# CHECK:         and     $6, $6, $15
# CHECK-NEXT:    lwc1    $f0, 0($6)

# CHECK:         and     $7, $7, $15
# CHECK-NEXT:    ldc1    $f2, 0($7)

# CHECK:         and     $8, $8, $15
# CHECK-NEXT:    ll      $4, 0($8)

# CHECK:         and     $9, $9, $15
# CHECK-NEXT:    lwl     $4, 0($9)

# CHECK:         and     $10, $10, $15
# CHECK-NEXT:    lwr     $4, 0($10)


# Check that loads where base register is $sp or $t8 (thread pointer register)
# are not masked.

# CHECK-NOT:     and
# CHECK:         lw      $4, 0($sp)
# CHECK-NOT:     and
# CHECK:         lw      $4, 0($24)



# Test that address-masking sandboxing is added before store instructions.

	.align	4
test3:
	.set	noreorder

        sb      $4, 0($1)
        nop
        sh      $4, 0($2)
        sw      $4, 0($3)
        swc1    $f0, 0($4)
        sdc1    $f2, 0($5)
        swl     $4, 0($6)
        swr     $4, 0($7)
        sc      $4, 0($8)

        sw      $4, 0($sp)
        sw      $4, 0($t8)

# CHECK-LABEL:   <test3>:

# CHECK:         and     $1, $1, $15
# CHECK-NEXT:    sb      $4, 0($1)

# Check that additional nop is inserted, to align mask and store to the next
# bundle.

# CHECK:         nop
# CHECK:         nop

# CHECK:         and     $2, $2, $15
# CHECK-NEXT:    sh      $4, 0($2)

# CHECK:         and     $3, $3, $15
# CHECK-NEXT:    sw      $4, 0($3)

# CHECK:         and     $4, $4, $15
# CHECK-NEXT:    swc1    $f0, 0($4)

# CHECK:         and     $5, $5, $15
# CHECK-NEXT:    sdc1    $f2, 0($5)

# CHECK:         and     $6, $6, $15
# CHECK-NEXT:    swl     $4, 0($6)

# CHECK:         and     $7, $7, $15
# CHECK-NEXT:    swr     $4, 0($7)

# CHECK:         and     $8, $8, $15
# CHECK-NEXT:    sc      $4, 0($8)


# Check that stores where base register is $sp or $t8 (thread pointer register)
# are not masked.

# CHECK-NOT:     and
# CHECK:         sw      $4, 0($sp)
# CHECK-NOT:     and
# CHECK:         sw      $4, 0($24)



# Test that address-masking sandboxing is added after instructions that change
# stack pointer.

	.align	4
test4:
	.set	noreorder

        addiu   $sp, $sp, 24
        nop
        addu    $sp, $sp, $1
        lw      $sp, 0($2)
        lw      $sp, 123($sp)
        sw      $sp, 123($sp)

# CHECK-LABEL:   <test4>:

# CHECK:         addiu   $sp, $sp, 24
# CHECK-NEXT:    and     $sp, $sp, $15

# Check that additional nop is inserted, to align instruction and mask to the
# next bundle.

# CHECK:         nop
# CHECK:         nop

# CHECK:         addu    $sp, $sp, $1
# CHECK-NEXT:    and     $sp, $sp, $15

# Since we next check sandboxing sequence which consists of 3 instructions,
# check that 2 additional nops are inserted, to align it to the next bundle.

# CHECK:         nop
# CHECK:         nop


# Check that for instructions that change stack-pointer and load from memory
# masks are added before and after the instruction.

# CHECK:         and     $2, $2, $15
# CHECK-NEXT:    lw      $sp, 0($2)
# CHECK-NEXT:    and     $sp, $sp, $15

# For loads where $sp is destination and base, check that mask is added after
# but not before.

# CHECK-NOT:     and
# CHECK:         lw      $sp, 123($sp)
# CHECK-NEXT:    and     $sp, $sp, $15

# For stores where $sp is destination and base, check that mask is added neither
# before nor after.

# CHECK-NOT:     and
# CHECK:         sw      $sp, 123($sp)
# CHECK-NOT:     and



# Test that call + branch delay is aligned at bundle end.  Test that mask is
# added before indirect calls.

	.align	4
test5:
	.set	noreorder

        jal func1
        addiu $4, $zero, 1

        nop
        bal func2
        addiu $4, $zero, 2

        nop
        nop
        bltzal $t1, func3
        addiu $4, $zero, 3

        nop
        nop
        nop
        bgezal $t2, func4
        addiu $4, $zero, 4

        jalr $t9
        addiu $4, $zero, 5


# CHECK:             nop
# CHECK-NEXT:        nop
# CHECK-LABEL:       <test5>:
# CHECK-NEXT:        jal
# CHECK-NEXT:        addiu   $4, $zero, 1

# CHECK-NEXT:        nop
# CHECK-NEXT:        nop
# CHECK-NEXT:        bal
# CHECK-NEXT:        addiu   $4, $zero, 2

# CHECK-NEXT:        nop
# CHECK-NEXT:        nop
# CHECK-NEXT:        bltzal
# CHECK-NEXT:        addiu   $4, $zero, 3

# CHECK-NEXT:        nop
# CHECK-NEXT:        nop
# CHECK-NEXT:        nop
# CHECK-NEXT:        nop

# CHECK-NEXT:        nop
# CHECK-NEXT:        nop
# CHECK-NEXT:        bgezal
# CHECK-NEXT:        addiu   $4, $zero, 4

# CHECK-NEXT:        nop
# CHECK-NEXT:        and     $25, $25, $14
# CHECK-NEXT:        jalr    $25
# CHECK-NEXT:        addiu   $4, $zero, 5



# Test that we can put non-dangerous loads and stores in branch delay slot.

	.align	4
test6:
	.set	noreorder

        jal func1
        sw      $4, 0($sp)

        bal func2
        lw      $5, 0($t8)

        jalr $t9
        sw      $sp, 0($sp)



# CHECK:             nop
# CHECK-NEXT:        nop
# CHECK-LABEL:       <test6>:
# CHECK-NEXT:        jal
# CHECK-NEXT:        sw      $4, 0($sp)

# CHECK-NEXT:        nop
# CHECK-NEXT:        nop
# CHECK-NEXT:        bal
# CHECK-NEXT:        lw      $5, 0($24)

# CHECK-NEXT:        nop
# CHECK-NEXT:        and     $25, $25, $14
# CHECK-NEXT:        jalr
# CHECK-NEXT:        sw      $sp, 0($sp)
