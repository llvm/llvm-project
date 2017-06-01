# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        di        $s8                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dext      $1, $2, 12, 12      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dextm     $1, $2, 21, 43      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dextu     $1, $2, 33, 16      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dins      $1, $2, 12, 12      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dinsm     $1, $2, 21, 43      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dinsu     $1, $2, 33, 16      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        drotr     $1,15               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        drotr     $1,$14,15           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        drotr32   $1,15               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        drotr32   $1,$14,15           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        drotrv    $1,$14,$15          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dsbh      $v1,$14             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dshd      $v0,$sp             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ei        $14                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mfhc1     $s8,$f24            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        mthc1     $zero,$f16          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        pause                         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        rotr      $1,15               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        rotr      $1,$14,15           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        rotrv     $1,$14,$15          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        seb       $25,$15             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        seh       $v1,$12             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        wsbh      $k1,$9              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        dins $2, $3, -1, 1            # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dins $2, $3, 64, 1            # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
        dinsm $2, $3, -1, 1           # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
        dinsm $2, $3, 32, 1           # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
        dinsm $2, $3, 31, 0           # CHECK: :[[@LINE]]:27: error: expected immediate in range 2 .. 64
        dinsm $2, $3, 31, 65          # CHECK: :[[@LINE]]:27: error: expected immediate in range 2 .. 64
        dinsu $2, $3, 31, 1           # CHECK: :[[@LINE]]:23: error: expected immediate in range 32 .. 63
        dinsu $2, $3, 64, 1           # CHECK: :[[@LINE]]:23: error: expected immediate in range 32 .. 63
        dinsu $2, $3, 63, 0           # CHECK: :[[@LINE]]:27: error: expected immediate in range 1 .. 32
        dinsu $2, $3, 32, 33          # CHECK: :[[@LINE]]:27: error: expected immediate in range 1 .. 32
