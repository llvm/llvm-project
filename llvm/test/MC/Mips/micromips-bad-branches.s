# RUN: not llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -mattr=+msa -triple=mips -mattr=+micromips 2>&1  | FileCheck %s
#
# CHECK: error: branch to misaligned address
# CHECK:        b -65535
# CHECK: error: branch target out of range
# CHECK:        b -65537
# CHECK: error: branch to misaligned address
# CHECK:        b 65535
# CHECK: error: branch target out of range
# CHECK:        b 65536

# CHECK: error: branch to misaligned address
# CHECK:        beq $1, $1, -65535
# CHECK: error: branch target out of range
# CHECK:        beq $1, $1, -65537
# CHECK: error: branch to misaligned address
# CHECK:        beq $1, $1, 65535
# CHECK: error: branch target out of range
# CHECK:        beq $1, $1, 65536

# CHECK: error: branch to misaligned address
# CHECK:        bne $1, $1, -65535
# CHECK: error: branch target out of range
# CHECK:        bne $1, $1, -65537
# CHECK: error: branch to misaligned address
# CHECK:        bne $1, $1, 65535
# CHECK: error: branch target out of range
# CHECK:        bne $1, $1, 65536

# CHECK: error: branch to misaligned address
# CHECK:        bal -65535
# CHECK: error: branch target out of range
# CHECK:        bal -65537
# CHECK: error: branch to misaligned address
# CHECK:        bal 65535
# CHECK: error: branch target out of range
# CHECK:        bal 65536

# CHECK: error: branch to misaligned address
# CHECK:        bgez $1, -65535
# CHECK: error: branch target out of range
# CHECK:        bgez $1, -65537
# CHECK: error: branch to misaligned address
# CHECK:        bgez $1, 65535
# CHECK: error: branch target out of range
# CHECK:        bgez $1, 65536

# CHECK: error: branch to misaligned address
# CHECK:        bgtz $1, -65535
# CHECK: error: branch target out of range
# CHECK:        bgtz $1, -65537
# CHECK: error: branch to misaligned address
# CHECK:        bgtz $1, 65535
# CHECK: error: branch target out of range
# CHECK:        bgtz $1, 65536

# CHECK: error: branch to misaligned address
# CHECK:        blez $1, -65535
# CHECK: error: branch target out of range
# CHECK:        blez $1, -65537
# CHECK: error: branch to misaligned address
# CHECK:        blez $1, 65535
# CHECK: error: branch target out of range
# CHECK:        blez $1, 65536

# CHECK: error: branch to misaligned address
# CHECK:        bltz $1, -65535
# CHECK: error: branch target out of range
# CHECK:        bltz $1, -65537
# CHECK: error: branch to misaligned address
# CHECK:        bltz $1, 65535
# CHECK: error: branch target out of range
# CHECK:        bltz $1, 65536

# CHECK: error: branch to misaligned address
# CHECK:        bgezal $1, -65535
# CHECK: error: branch target out of range
# CHECK:        bgezal $1, -65537
# CHECK: error: branch to misaligned address
# CHECK:        bgezal $1, 65535
# CHECK: error: branch target out of range
# CHECK:        bgezal $1, 65536

# CHECK: error: branch to misaligned address
# CHECK:        bltzal $1, -65535
# CHECK: error: branch target out of range
# CHECK:        bltzal $1, -65537
# CHECK: error: branch to misaligned address
# CHECK:        bltzal $1, 65535
# CHECK: error: branch target out of range
# CHECK:        bltzal $1, 65536

# CHECK: error: branch to misaligned address
# CHECK:        bc1f -65535
# CHECK: error: branch target out of range
# CHECK:        bc1f -65537
# CHECK: error: branch to misaligned address
# CHECK:        bc1f 65535
# CHECK: error: branch target out of range
# CHECK:        bc1f 65536

# CHECK: error: branch to misaligned address
# CHECK:        bc1f $fcc0, -65535
# CHECK: error: branch target out of range
# CHECK:        bc1f $fcc0, -65537
# CHECK: error: branch to misaligned address
# CHECK:        bc1f $fcc0, 65535
# CHECK: error: branch target out of range
# CHECK:        bc1f $fcc0, 65536

# CHECK: error: branch to misaligned address
# CHECK:        bc1t -65535
# CHECK: error: branch target out of range
# CHECK:        bc1t -65537
# CHECK: error: branch to misaligned address
# CHECK:        bc1t 65535
# CHECK: error: branch target out of range
# CHECK:        bc1t 65536

# CHECK: error: branch to misaligned address
# CHECK:        bc1t $fcc0, -65535
# CHECK: error: branch target out of range
# CHECK:        bc1t $fcc0, -65537
# CHECK: error: branch to misaligned address
# CHECK:        bc1t $fcc0, 65535
# CHECK: error: branch target out of range
# CHECK:        bc1t $fcc0, 65536

# CHECK: error: branch to misaligned address
# CHECK:        beqz16 $6, 31
# CHECK: error: branch target out of range
# CHECK:        beqz16 $6, 130

        b -65535
        b -65536
        b -65537
        b 65534
        b 65535
        b 65536

        beq $1, $1, -65535
        beq $1, $1, -65536
        beq $1, $1, -65537
        beq $1, $1, 65534
        beq $1, $1, 65535
        beq $1, $1, 65536

        bne $1, $1, -65535
        bne $1, $1, -65536
        bne $1, $1, -65537
        bne $1, $1, 65534
        bne $1, $1, 65535
        bne $1, $1, 65536

        bal -65535
        bal -65536
        bal -65537
        bal 65534
        bal 65535
        bal 65536

        bgez $1, -65535
        bgez $1, -65536
        bgez $1, -65537
        bgez $1, 65534
        bgez $1, 65535
        bgez $1, 65536

        bgtz $1, -65535
        bgtz $1, -65536
        bgtz $1, -65537
        bgtz $1, 65534
        bgtz $1, 65535
        bgtz $1, 65536

        blez $1, -65535
        blez $1, -65536
        blez $1, -65537
        blez $1, 65534
        blez $1, 65535
        blez $1, 65536

        bltz $1, -65535
        bltz $1, -65536
        bltz $1, -65537
        bltz $1, 65534
        bltz $1, 65535
        bltz $1, 65536

        bgezal $1, -65535
        bgezal $1, -65536
        bgezal $1, -65537
        bgezal $1, 65534
        bgezal $1, 65535
        bgezal $1, 65536

        bltzal $1, -65535
        bltzal $1, -65536
        bltzal $1, -65537
        bltzal $1, 65534
        bltzal $1, 65535
        bltzal $1, 65536

        bc1f -65535
        bc1f -65536
        bc1f -65537
        bc1f 65534
        bc1f 65535
        bc1f 65536

        bc1f $fcc0, -65535
        bc1f $fcc0, -65536
        bc1f $fcc0, -65537
        bc1f $fcc0, 65534
        bc1f $fcc0, 65535
        bc1f $fcc0, 65536

        bc1t -65535
        bc1t -65536
        bc1t -65537
        bc1t 65534
        bc1t 65535
        bc1t 65536

        bc1t $fcc0, -65535
        bc1t $fcc0, -65536
        bc1t $fcc0, -65537
        bc1t $fcc0, 65534
        bc1t $fcc0, 65535
        bc1t $fcc0, 65536

        beqz16 $6, 31
        beqz16 $6, 130
