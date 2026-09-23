# Instructions that are invalid on R5900
#
# R5900 is a MIPS III-based processor with specific limitations:
# - No 64-bit multiply/divide (DMULT/DMULTU/DDIV/DDIVU)
# - No LL/SC atomic instructions
# - No COP3 instructions (LWC3, SWC3, LDC3, SDC3)
# - No ROUND/TRUNC/CEIL/FLOOR.W.S (uses CVT.W.S instead)
# - No double-precision FPU (single float only)
#
# RUN: not llvm-mc %s -triple=mips64el-unknown-linux -mcpu=r5900 2>%t1
# RUN: FileCheck %s < %t1

        .set noat

# =============================================================================
# MIPS3 64-bit multiply/divide instructions that R5900 does NOT support
# =============================================================================

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dmult   $4, $5

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dmultu  $6, $7

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ddiv    $zero, $8, $9

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ddivu   $zero, $10, $11

# =============================================================================
# LL/SC atomic instructions that R5900 does NOT support
# =============================================================================

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ll      $4, 0($5)

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sc      $4, 0($5)

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        lld     $4, 0($5)

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        scd     $4, 0($5)

# =============================================================================
# COP3 instructions that R5900 does NOT support
# =============================================================================

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        lwc3    $4, 0($5)

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        swc3    $4, 0($5)

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ldc3    $4, 0($5)

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sdc3    $4, 0($5)

# =============================================================================
# Single-precision FP compare instructions that R5900 does NOT support
# (R5900 only supports c.f.s, c.eq.s, c.olt.s, c.ole.s)
# =============================================================================

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.un.s  $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.ueq.s $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.ult.s $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.ule.s $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.sf.s  $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.ngle.s $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.seq.s $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.ngl.s $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.lt.s  $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.nge.s $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.le.s  $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.ngt.s $f4, $f6

# =============================================================================
# FPU rounding instructions that R5900 does NOT support
# (R5900 uses CVT.W.S which always truncates toward zero)
# =============================================================================

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        round.w.s $f4, $f5

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        trunc.w.s $f6, $f7

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ceil.w.s  $f8, $f9

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        floor.w.s $f10, $f11

# =============================================================================
# Double-precision FPU instructions that R5900 does NOT support
# (R5900 has single-precision FPU only)
# =============================================================================

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        add.d   $f4, $f6, $f8

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sub.d   $f4, $f6, $f8

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mul.d   $f4, $f6, $f8

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        div.d   $f4, $f6, $f8

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        abs.d   $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        neg.d   $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sqrt.d  $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mov.d   $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.s.d $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.d.s $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.d.w $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        cvt.w.d $f4, $f6

# 64-bit FP load/store
# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ldc1    $f4, 0($5)

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        sdc1    $f4, 0($5)

# Double-precision comparisons
# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.eq.d  $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.lt.d  $f4, $f6

# CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        c.le.d  $f4, $f6
