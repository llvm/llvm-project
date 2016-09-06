# RUN: llvm-mc %s -triple=mips-unknown-unknown -show-encoding -mcpu=mips32r2 | \
# RUN: FileCheck -check-prefix=CHECK32  %s
# RUN: llvm-mc %s -triple=mips64-unknown-unknown -show-encoding -mcpu=mips64r2 \
# RUN: | FileCheck -check-prefix=CHECK64  %s

# CHECK32:    break                      # encoding: [0x00,0x00,0x00,0x0d]
# CHECK32:    break   7                  # encoding: [0x00,0x07,0x00,0x0d]
# CHECK32:    break   7, 5               # encoding: [0x00,0x07,0x01,0x4d]
# CHECK32:    eret                       # encoding: [0x42,0x00,0x00,0x18]
# CHECK32:    deret                      # encoding: [0x42,0x00,0x00,0x1f]
# CHECK32:    di                         # encoding: [0x41,0x60,0x60,0x00]
# CHECK32:    di                         # encoding: [0x41,0x60,0x60,0x00]
# CHECK32:    di      $10                # encoding: [0x41,0x6a,0x60,0x00]
# CHECK32:    ei                         # encoding: [0x41,0x60,0x60,0x20]
# CHECK32:    ei                         # encoding: [0x41,0x60,0x60,0x20]
# CHECK32:    ei      $10                # encoding: [0x41,0x6a,0x60,0x20]
# CHECK32:    wait                       # encoding: [0x42,0x00,0x00,0x20]
# CHECK32:    teq     $zero, $3          # encoding: [0x00,0x03,0x00,0x34]
# CHECK32:    teq     $zero, $3, 1       # encoding: [0x00,0x03,0x00,0x74]
# CHECK32:    teqi    $3, 1              # encoding: [0x04,0x6c,0x00,0x01]
# CHECK32:    tge     $zero, $3          # encoding: [0x00,0x03,0x00,0x30]
# CHECK32:    tge     $zero, $3, 3       # encoding: [0x00,0x03,0x00,0xf0]
# CHECK32:    tgei    $3, 3              # encoding: [0x04,0x68,0x00,0x03]
# CHECK32:    tgeu    $zero, $3          # encoding: [0x00,0x03,0x00,0x31]
# CHECK32:    tgeu    $zero, $3, 7       # encoding: [0x00,0x03,0x01,0xf1]
# CHECK32:    tgeiu   $3, 7              # encoding: [0x04,0x69,0x00,0x07]
# CHECK32:    tlt     $zero, $3          # encoding: [0x00,0x03,0x00,0x32]
# CHECK32:    tlt     $zero, $3, 31      # encoding: [0x00,0x03,0x07,0xf2]
# CHECK32:    tlti    $3, 31             # encoding: [0x04,0x6a,0x00,0x1f]
# CHECK32:    tltu    $zero, $3          # encoding: [0x00,0x03,0x00,0x33]
# CHECK32:    tltu    $zero, $3, 255     # encoding: [0x00,0x03,0x3f,0xf3]
# CHECK32:    tltiu   $3, 255            # encoding: [0x04,0x6b,0x00,0xff]
# CHECK32:    tne     $zero, $3          # encoding: [0x00,0x03,0x00,0x36]
# CHECK32:    tne     $zero, $3, 1023    # encoding: [0x00,0x03,0xff,0xf6]
# CHECK32:    tnei    $3, 1023           # encoding: [0x04,0x6e,0x03,0xff]

# CHECK64:    break                      # encoding: [0x00,0x00,0x00,0x0d]
# CHECK64:    break   7                  # encoding: [0x00,0x07,0x00,0x0d]
# CHECK64:    break   7, 5               # encoding: [0x00,0x07,0x01,0x4d]
# CHECK64:    eret                       # encoding: [0x42,0x00,0x00,0x18]
# CHECK64:    deret                      # encoding: [0x42,0x00,0x00,0x1f]
# CHECK64:    di                         # encoding: [0x41,0x60,0x60,0x00]
# CHECK64:    di                         # encoding: [0x41,0x60,0x60,0x00]
# CHECK64:    di      $10                # encoding: [0x41,0x6a,0x60,0x00]
# CHECK64:    ei                         # encoding: [0x41,0x60,0x60,0x20]
# CHECK64:    ei                         # encoding: [0x41,0x60,0x60,0x20]
# CHECK64:    ei      $10                # encoding: [0x41,0x6a,0x60,0x20]
# CHECK64:    wait                       # encoding: [0x42,0x00,0x00,0x20]
# CHECK64:    teq     $zero, $3          # encoding: [0x00,0x03,0x00,0x34]
# CHECK64:    teq     $zero, $3, 1       # encoding: [0x00,0x03,0x00,0x74]
# CHECK64:    teqi    $3, 1              # encoding: [0x04,0x6c,0x00,0x01]
# CHECK64:    tge     $zero, $3          # encoding: [0x00,0x03,0x00,0x30]
# CHECK64:    tge     $zero, $3, 3       # encoding: [0x00,0x03,0x00,0xf0]
# CHECK64:    tgei    $3, 3              # encoding: [0x04,0x68,0x00,0x03]
# CHECK64:    tgeu    $zero, $3          # encoding: [0x00,0x03,0x00,0x31]
# CHECK64:    tgeu    $zero, $3, 7       # encoding: [0x00,0x03,0x01,0xf1]
# CHECK64:    tgeiu   $3, 7              # encoding: [0x04,0x69,0x00,0x07]
# CHECK64:    tlt     $zero, $3          # encoding: [0x00,0x03,0x00,0x32]
# CHECK64:    tlt     $zero, $3, 31      # encoding: [0x00,0x03,0x07,0xf2]
# CHECK64:    tlti    $3, 31             # encoding: [0x04,0x6a,0x00,0x1f]
# CHECK64:    tltu    $zero, $3          # encoding: [0x00,0x03,0x00,0x33]
# CHECK64:    tltu    $zero, $3, 255     # encoding: [0x00,0x03,0x3f,0xf3]
# CHECK64:    tltiu   $3, 255            # encoding: [0x04,0x6b,0x00,0xff]
# CHECK64:    tne     $zero, $3          # encoding: [0x00,0x03,0x00,0x36]
# CHECK64:    tne     $zero, $3, 1023    # encoding: [0x00,0x03,0xff,0xf6]
# CHECK64:    tnei    $3, 1023           # encoding: [0x04,0x6e,0x03,0xff]

    break
    break 7
    break 7,5
    eret
    deret
    di
    di  $0
    di  $10

    ei
    ei  $0
    ei  $10

    wait

    teq   $0,$3
    teq   $0,$3,1
    teqi  $3,1
    tge   $0,$3
    tge   $0,$3,3
    tgei  $3,3
    tgeu  $0,$3
    tgeu  $0,$3,7
    tgeiu $3,7
    tlt   $0,$3
    tlt   $0,$3,31
    tlti  $3,31
    tltu  $0,$3
    tltu  $0,$3,255
    tltiu $3,255
    tne   $0,$3
    tne   $0,$3,1023
    tnei  $3,1023
