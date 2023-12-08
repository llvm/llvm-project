# RUN: not llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -triple=mips 2>&1 | FileCheck %s
#
# CHECK: error: branch to misaligned address
# CHECK:        b -131069
# CHECK: error: branch to misaligned address
# CHECK:        b -131070
# CHECK: error: branch to misaligned address
# CHECK:        b -131071
# CHECK: error: branch target out of range
# CHECK:        b -131073
# CHECK: error: branch to misaligned address
# CHECK:        b 131069
# CHECK: error: branch to misaligned address
# CHECK:        b 131070
# CHECK: error: branch to misaligned address
# CHECK:        b 131071
# CHECK: error: branch target out of range
# CHECK:        b 131072

# CHECK: error: branch to misaligned address
# CHECK:        beq $1, $1, -131069
# CHECK: error: branch to misaligned address
# CHECK:        beq $1, $1, -131070
# CHECK: error: branch to misaligned address
# CHECK:        beq $1, $1, -131071
# CHECK: error: branch target out of range
# CHECK:        beq $1, $1, -131073
# CHECK: error: branch to misaligned address
# CHECK:        beq $1, $1, 131069
# CHECK: error: branch to misaligned address
# CHECK:        beq $1, $1, 131070
# CHECK: error: branch to misaligned address
# CHECK:        beq $1, $1, 131071
# CHECK: error: branch target out of range
# CHECK:        beq $1, $1, 131072

# CHECK: error: branch to misaligned address
# CHECK:        bne $1, $1, -131069
# CHECK: error: branch to misaligned address
# CHECK:        bne $1, $1, -131070
# CHECK: error: branch to misaligned address
# CHECK:        bne $1, $1, -131071
# CHECK: error: branch target out of range
# CHECK:        bne $1, $1, -131073
# CHECK: error: branch to misaligned address
# CHECK:        bne $1, $1, 131069
# CHECK: error: branch to misaligned address
# CHECK:        bne $1, $1, 131070
# CHECK: error: branch to misaligned address
# CHECK:        bne $1, $1, 131071
# CHECK: error: branch target out of range
# CHECK:        bne $1, $1, 131072

# CHECK: error: branch to misaligned address
# CHECK:        bal -131069
# CHECK: error: branch to misaligned address
# CHECK:        bal -131070
# CHECK: error: branch to misaligned address
# CHECK:        bal -131071
# CHECK: error: branch target out of range
# CHECK:        bal -131073
# CHECK: error: branch to misaligned address
# CHECK:        bal 131069
# CHECK: error: branch to misaligned address
# CHECK:        bal 131070
# CHECK: error: branch to misaligned address
# CHECK:        bal 131071
# CHECK: error: branch target out of range
# CHECK:        bal 131072

# CHECK: error: branch to misaligned address
# CHECK:        bgez $1, -131069
# CHECK: error: branch to misaligned address
# CHECK:        bgez $1, -131070
# CHECK: error: branch to misaligned address
# CHECK:        bgez $1, -131071
# CHECK: error: branch target out of range
# CHECK:        bgez $1, -131073
# CHECK: error: branch to misaligned address
# CHECK:        bgez $1, 131069
# CHECK: error: branch to misaligned address
# CHECK:        bgez $1, 131070
# CHECK: error: branch to misaligned address
# CHECK:        bgez $1, 131071
# CHECK: error: branch target out of range
# CHECK:        bgez $1, 131072

# CHECK: error: branch to misaligned address
# CHECK:        bgtz $1, -131069
# CHECK: error: branch to misaligned address
# CHECK:        bgtz $1, -131070
# CHECK: error: branch to misaligned address
# CHECK:        bgtz $1, -131071
# CHECK: error: branch target out of range
# CHECK:        bgtz $1, -131073
# CHECK: error: branch to misaligned address
# CHECK:        bgtz $1, 131069
# CHECK: error: branch to misaligned address
# CHECK:        bgtz $1, 131070
# CHECK: error: branch to misaligned address
# CHECK:        bgtz $1, 131071
# CHECK: error: branch target out of range
# CHECK:        bgtz $1, 131072

# CHECK: error: branch to misaligned address
# CHECK:        blez $1, -131069
# CHECK: error: branch to misaligned address
# CHECK:        blez $1, -131070
# CHECK: error: branch to misaligned address
# CHECK:        blez $1, -131071
# CHECK: error: branch target out of range
# CHECK:        blez $1, -131073
# CHECK: error: branch to misaligned address
# CHECK:        blez $1, 131069
# CHECK: error: branch to misaligned address
# CHECK:        blez $1, 131070
# CHECK: error: branch to misaligned address
# CHECK:        blez $1, 131071
# CHECK: error: branch target out of range
# CHECK:        blez $1, 131072

# CHECK: error: branch to misaligned address
# CHECK:        bltz $1, -131069
# CHECK: error: branch to misaligned address
# CHECK:        bltz $1, -131070
# CHECK: error: branch to misaligned address
# CHECK:        bltz $1, -131071
# CHECK: error: branch target out of range
# CHECK:        bltz $1, -131073
# CHECK: error: branch to misaligned address
# CHECK:        bltz $1, 131069
# CHECK: error: branch to misaligned address
# CHECK:        bltz $1, 131070
# CHECK: error: branch to misaligned address
# CHECK:        bltz $1, 131071
# CHECK: error: branch target out of range
# CHECK:        bltz $1, 131072

# CHECK: error: branch to misaligned address
# CHECK:        bgezal $1, -131069
# CHECK: error: branch to misaligned address
# CHECK:        bgezal $1, -131070
# CHECK: error: branch to misaligned address
# CHECK:        bgezal $1, -131071
# CHECK: error: branch target out of range
# CHECK:        bgezal $1, -131073
# CHECK: error: branch to misaligned address
# CHECK:        bgezal $1, 131069
# CHECK: error: branch to misaligned address
# CHECK:        bgezal $1, 131070
# CHECK: error: branch to misaligned address
# CHECK:        bgezal $1, 131071
# CHECK: error: branch target out of range
# CHECK:        bgezal $1, 131072

# CHECK: error: branch to misaligned address
# CHECK:        bltzal $1, -131069
# CHECK: error: branch to misaligned address
# CHECK:        bltzal $1, -131070
# CHECK: error: branch to misaligned address
# CHECK:        bltzal $1, -131071
# CHECK: error: branch target out of range
# CHECK:        bltzal $1, -131073
# CHECK: error: branch to misaligned address
# CHECK:        bltzal $1, 131069
# CHECK: error: branch to misaligned address
# CHECK:        bltzal $1, 131070
# CHECK: error: branch to misaligned address
# CHECK:        bltzal $1, 131071
# CHECK: error: branch target out of range
# CHECK:        bltzal $1, 131072

# CHECK: error: branch to misaligned address
# CHECK:        bc1f -131069
# CHECK: error: branch to misaligned address
# CHECK:        bc1f -131070
# CHECK: error: branch to misaligned address
# CHECK:        bc1f -131071
# CHECK: error: branch target out of range
# CHECK:        bc1f -131073
# CHECK: error: branch to misaligned address
# CHECK:        bc1f 131069
# CHECK: error: branch to misaligned address
# CHECK:        bc1f 131070
# CHECK: error: branch to misaligned address
# CHECK:        bc1f 131071
# CHECK: error: branch target out of range
# CHECK:        bc1f 131072

# CHECK: error: branch to misaligned address
# CHECK:        bc1f $fcc0, -131069
# CHECK: error: branch to misaligned address
# CHECK:        bc1f $fcc0, -131070
# CHECK: error: branch to misaligned address
# CHECK:        bc1f $fcc0, -131071
# CHECK: error: branch target out of range
# CHECK:        bc1f $fcc0, -131073
# CHECK: error: branch to misaligned address
# CHECK:        bc1f $fcc0, 131069
# CHECK: error: branch to misaligned address
# CHECK:        bc1f $fcc0, 131070
# CHECK: error: branch to misaligned address
# CHECK:        bc1f $fcc0, 131071
# CHECK: error: branch target out of range
# CHECK:        bc1f $fcc0, 131072

# CHECK: error: branch to misaligned address
# CHECK:        bc1t -131069
# CHECK: error: branch to misaligned address
# CHECK:        bc1t -131070
# CHECK: error: branch to misaligned address
# CHECK:        bc1t -131071
# CHECK: error: branch target out of range
# CHECK:        bc1t -131073
# CHECK: error: branch to misaligned address
# CHECK:        bc1t 131069
# CHECK: error: branch to misaligned address
# CHECK:        bc1t 131070
# CHECK: error: branch to misaligned address
# CHECK:        bc1t 131071
# CHECK: error: branch target out of range
# CHECK:        bc1t 131072

# CHECK: error: branch to misaligned address
# CHECK:        bc1t $fcc0, -131069
# CHECK: error: branch to misaligned address
# CHECK:        bc1t $fcc0, -131070
# CHECK: error: branch to misaligned address
# CHECK:        bc1t $fcc0, -131071
# CHECK: error: branch target out of range
# CHECK:        bc1t $fcc0, -131073
# CHECK: error: branch to misaligned address
# CHECK:        bc1t $fcc0, 131069
# CHECK: error: branch to misaligned address
# CHECK:        bc1t $fcc0, 131070
# CHECK: error: branch to misaligned address
# CHECK:        bc1t $fcc0, 131071
# CHECK: error: branch target out of range
# CHECK:        bc1t $fcc0, 131072

.text
.set noat
  b -131068
  b -131069
  b -131070
  b -131071
  b -131072
  b -131073
  b 131068
  b 131069
  b 131070
  b 131071
  b 131072

  beq $1, $1, -131068
  beq $1, $1, -131069
  beq $1, $1, -131070
  beq $1, $1, -131071
  beq $1, $1, -131072
  beq $1, $1, -131073
  beq $1, $1, 131068
  beq $1, $1, 131069
  beq $1, $1, 131070
  beq $1, $1, 131071
  beq $1, $1, 131072

  bne $1, $1, -131068
  bne $1, $1, -131069
  bne $1, $1, -131070
  bne $1, $1, -131071
  bne $1, $1, -131072
  bne $1, $1, -131073
  bne $1, $1, 131068
  bne $1, $1, 131069
  bne $1, $1, 131070
  bne $1, $1, 131071
  bne $1, $1, 131072

  bal -131068
  bal -131069
  bal -131070
  bal -131071
  bal -131072
  bal -131073
  bal 131068
  bal 131069
  bal 131070
  bal 131071
  bal 131072

  bgez $1, -131068
  bgez $1, -131069
  bgez $1, -131070
  bgez $1, -131071
  bgez $1, -131072
  bgez $1, -131073
  bgez $1, 131068
  bgez $1, 131069
  bgez $1, 131070
  bgez $1, 131071
  bgez $1, 131072

  bgtz $1, -131068
  bgtz $1, -131069
  bgtz $1, -131070
  bgtz $1, -131071
  bgtz $1, -131072
  bgtz $1, -131073
  bgtz $1, 131068
  bgtz $1, 131069
  bgtz $1, 131070
  bgtz $1, 131071
  bgtz $1, 131072

  blez $1, -131068
  blez $1, -131069
  blez $1, -131070
  blez $1, -131071
  blez $1, -131072
  blez $1, -131073
  blez $1, 131068
  blez $1, 131069
  blez $1, 131070
  blez $1, 131071
  blez $1, 131072

  bltz $1, -131068
  bltz $1, -131069
  bltz $1, -131070
  bltz $1, -131071
  bltz $1, -131072
  bltz $1, -131073
  bltz $1, 131068
  bltz $1, 131069
  bltz $1, 131070
  bltz $1, 131071
  bltz $1, 131072

  bgezal $1, -131068
  bgezal $1, -131069
  bgezal $1, -131070
  bgezal $1, -131071
  bgezal $1, -131072
  bgezal $1, -131073
  bgezal $1, 131068
  bgezal $1, 131069
  bgezal $1, 131070
  bgezal $1, 131071
  bgezal $1, 131072

  bltzal $1, -131068
  bltzal $1, -131069
  bltzal $1, -131070
  bltzal $1, -131071
  bltzal $1, -131072
  bltzal $1, -131073
  bltzal $1, 131068
  bltzal $1, 131069
  bltzal $1, 131070
  bltzal $1, 131071
  bltzal $1, 131072

  bc1f -131068
  bc1f -131069
  bc1f -131070
  bc1f -131071
  bc1f -131072
  bc1f -131073
  bc1f 131068
  bc1f 131069
  bc1f 131070
  bc1f 131071
  bc1f 131072

  bc1f $fcc0, -131068
  bc1f $fcc0, -131069
  bc1f $fcc0, -131070
  bc1f $fcc0, -131071
  bc1f $fcc0, -131072
  bc1f $fcc0, -131073
  bc1f $fcc0, 131068
  bc1f $fcc0, 131069
  bc1f $fcc0, 131070
  bc1f $fcc0, 131071
  bc1f $fcc0, 131072

  bc1t -131068
  bc1t -131069
  bc1t -131070
  bc1t -131071
  bc1t -131072
  bc1t -131073
  bc1t 131068
  bc1t 131069
  bc1t 131070
  bc1t 131071
  bc1t 131072

  bc1t $fcc0, -131068
  bc1t $fcc0, -131069
  bc1t $fcc0, -131070
  bc1t $fcc0, -131071
  bc1t $fcc0, -131072
  bc1t $fcc0, -131073
  bc1t $fcc0, 131068
  bc1t $fcc0, 131069
  bc1t $fcc0, 131070
  bc1t $fcc0, 131071
  bc1t $fcc0, 131072
