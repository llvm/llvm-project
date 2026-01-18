# Instructions that are invalid on R5900
#
# R5900 is a MIPS III-based processor with specific limitations:
# - No 64-bit multiply/divide (DMULT/DMULTU/DDIV/DDIVU)
# - No LL/SC atomic instructions
# - No COP3 instructions (LWC3, SWC3, LDC3, SDC3)
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
