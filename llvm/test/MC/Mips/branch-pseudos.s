# RUN: llvm-mc %s -triple=mips -mcpu=mips32 -show-encoding | FileCheck %s
# RUN: llvm-mc %s -triple=mips -mcpu=mips32 2>&1 | \
# RUN:   FileCheck %s --check-prefix=WARNING

  .text
local_label:
  blt $7, $8, local_label
# CHECK: slt  $1, $7, $8       # encoding: [0x00,0xe8,0x08,0x2a]
# CHECK: bnez $1, local_label  # encoding: [0x14,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  blt $7, $8, global_label
# CHECK: slt  $1, $7, $8       # encoding: [0x00,0xe8,0x08,0x2a]
# CHECK: bnez $1, global_label # encoding: [0x14,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  blt $7, $0, local_label
# CHECK: bltz $7, local_label  # encoding: [0x04,0xe0,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  blt $0, $8, local_label
# CHECK: bgtz $8, local_label  # encoding: [0x1d,0x00,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  blt $0, $0, local_label
# CHECK: bltz $zero, local_label # encoding: [0x04,0x00,A,A]
# CHECK:                         #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop

  bltu $7, $8, local_label
# CHECK: sltu $1, $7, $8       # encoding: [0x00,0xe8,0x08,0x2b]
# CHECK: bnez $1, local_label  # encoding: [0x14,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bltu $7, $8, global_label
# CHECK: sltu $1, $7, $8       # encoding: [0x00,0xe8,0x08,0x2b]
# CHECK: bnez $1, global_label # encoding: [0x14,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bltu $7, $0, local_label
# CHECK: nop
  bltu $0, $8, local_label
# CHECK: bnez $8, local_label  # encoding: [0x15,0x00,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bltu $0, $0, local_label
# CHECK: nop

  ble $7, $8, local_label
# CHECK: slt  $1, $8, $7       # encoding: [0x01,0x07,0x08,0x2a]
# CHECK: beqz $1, local_label  # encoding: [0x10,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  ble $7, $8, global_label
# CHECK: slt  $1, $8, $7       # encoding: [0x01,0x07,0x08,0x2a]
# CHECK: beqz $1, global_label # encoding: [0x10,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  ble $7, $0, local_label
# CHECK: blez $7, local_label  # encoding: [0x18,0xe0,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  ble $0, $8, local_label
# CHECK: bgez $8, local_label  # encoding: [0x05,0x01,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  ble $0, $0, local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: blez $zero, local_label # encoding: [0x18,0x00,A,A]
# CHECK:                         #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop

  bleu $7, $8, local_label
# CHECK: sltu $1, $8, $7       # encoding: [0x01,0x07,0x08,0x2b]
# CHECK: beqz $1, local_label  # encoding: [0x10,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bleu $7, $8, global_label
# CHECK: sltu $1, $8, $7       # encoding: [0x01,0x07,0x08,0x2b]
# CHECK: beqz $1, global_label # encoding: [0x10,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bleu $7, $0, local_label
# CHECK: beqz $7, local_label  # encoding: [0x10,0xe0,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bleu $0, $8, local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: b  local_label        # encoding: [0x10,0x00,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bleu $0, $0, local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: b  local_label        # encoding: [0x10,0x00,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop

  bge $7, $8, local_label
# CHECK: slt  $1, $7, $8       # encoding: [0x00,0xe8,0x08,0x2a]
# CHECK: beqz $1, local_label  # encoding: [0x10,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bge $7, $8, global_label
# CHECK: slt  $1, $7, $8       # encoding: [0x00,0xe8,0x08,0x2a]
# CHECK: beqz $1, global_label # encoding: [0x10,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bge $7, $0, local_label
# CHECK: bgez $7, local_label  # encoding: [0x04,0xe1,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bge $0, $8, local_label
# CHECK: blez $8, local_label  # encoding: [0x19,0x00,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bge $0, $0, local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: bgez $zero, local_label # encoding: [0x04,0x01,A,A]
# CHECK:                         #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop

  bgeu $7, $8, local_label
# CHECK: sltu $1, $7, $8       # encoding: [0x00,0xe8,0x08,0x2b]
# CHECK: beqz $1, local_label  # encoding: [0x10,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgeu $7, $8, global_label
# CHECK: sltu $1, $7, $8       # encoding: [0x00,0xe8,0x08,0x2b]
# CHECK: beqz $1, global_label # encoding: [0x10,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgeu $7, $0, local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: b  local_label        # encoding: [0x10,0x00,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgeu $0, $8, local_label
# CHECK: beqz $8, local_label  # encoding: [0x11,0x00,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgeu $0, $0, local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: b  local_label        # encoding: [0x10,0x00,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop

  bgt $7, $8, local_label
# CHECK: slt  $1, $8, $7       # encoding: [0x01,0x07,0x08,0x2a]
# CHECK: bnez $1, local_label  # encoding: [0x14,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgt $7, $8, global_label
# CHECK: slt  $1, $8, $7       # encoding: [0x01,0x07,0x08,0x2a]
# CHECK: bnez $1, global_label # encoding: [0x14,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgt $7, $0, local_label
# CHECK: bgtz $7, local_label  # encoding: [0x1c,0xe0,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgt $0, $8, local_label
# CHECK: bltz $8, local_label  # encoding: [0x05,0x00,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgt $0, $0, local_label
# CHECK: bgtz  $zero, local_label # encoding: [0x1c,0x00,A,A]
# CHECK:                          #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop

  bgtu $7, $8, local_label
# CHECK: sltu $1, $8, $7       # encoding: [0x01,0x07,0x08,0x2b]
# CHECK: bnez $1, local_label  # encoding: [0x14,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgtu $7, $8, global_label
# CHECK: sltu $1, $8, $7       # encoding: [0x01,0x07,0x08,0x2b]
# CHECK: bnez $1, global_label # encoding: [0x14,0x20,A,A]
# CHECK:                       #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgtu $7, $0, local_label
# CHECK: bnez $7, local_label  # encoding: [0x14,0xe0,A,A]
# CHECK:                       #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop
  bgtu $0, $8, local_label
# CHECK: nop
  bgtu $0, $0, local_label
# CHECK: bnez $zero, local_label # encoding: [0x14,0x00,A,A]
# CHECK:                         #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop

  bltl $7,$8,local_label
# CHECK: slt $1, $7, $8                 # encoding: [0x00,0xe8,0x08,0x2a]
# CHECK: bnezl $1, local_label          # encoding: [0x54,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bltl $7,$8,global_label
# CHECK: slt $1, $7, $8                 # encoding: [0x00,0xe8,0x08,0x2a]
# CHECK: bnezl $1, global_label         # encoding: [0x54,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bltl $7,$0,local_label
# CHECK: bltz $7, local_label           # encoding: [0x04,0xe0,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bltl $0,$8,local_label
# CHECK: bgtz $8, local_label           # encoding: [0x1d,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bltl $0,$0,local_label
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]

  blel $7,$8,local_label
# CHECK: slt $1, $8, $7                 # encoding: [0x01,0x07,0x08,0x2a]
# CHECK: beqzl $1, local_label          # encoding: [0x50,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  blel $7,$8,global_label
# CHECK: slt $1, $8, $7                 # encoding: [0x01,0x07,0x08,0x2a]
# CHECK: beqzl $1, global_label         # encoding: [0x50,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  blel $7,$0,local_label
# CHECK: blez $7, local_label           # encoding: [0x18,0xe0,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  blel $0,$8,local_label
# CHECK: bgez $8, local_label           # encoding: [0x05,0x01,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  blel $0,$0,local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: b local_label                  # encoding: [0x10,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]

  bgel $7,$8,local_label
# CHECK: slt $1, $7, $8                 # encoding: [0x00,0xe8,0x08,0x2a]
# CHECK: beqzl $1, local_label          # encoding: [0x50,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgel $7,$8,global_label
# CHECK: slt $1, $7, $8                 # encoding: [0x00,0xe8,0x08,0x2a]
# CHECK: beqzl $1, global_label         # encoding: [0x50,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgel $7,$0,local_label
# CHECK: bgez $7, local_label           # encoding: [0x04,0xe1,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgel $0,$8,local_label
# CHECK: blez $8, local_label           # encoding: [0x19,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgel $0,$0,local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: b local_label                  # encoding: [0x10,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]

  bgtl $7,$8,local_label
# CHECK: slt $1, $8, $7                 # encoding: [0x01,0x07,0x08,0x2a]
# CHECK: bnezl $1, local_label          # encoding: [0x54,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgtl $7,$8,global_label
# CHECK: slt $1, $8, $7                 # encoding: [0x01,0x07,0x08,0x2a]
# CHECK: bnezl $1, global_label         # encoding: [0x54,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgtl $7,$0,local_label
# CHECK: bgtz $7, local_label           # encoding: [0x1c,0xe0,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgtl $0,$8,local_label
# CHECK: bltz $8, local_label           # encoding: [0x05,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgtl $0,$0,local_label
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]

  bltul $7,$8,local_label
# CHECK: sltu $1, $7, $8                # encoding: [0x00,0xe8,0x08,0x2b]
# CHECK: bnezl $1, local_label          # encoding: [0x54,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bltul $7,$8,global_label
# CHECK: sltu $1, $7, $8                # encoding: [0x00,0xe8,0x08,0x2b]
# CHECK: bnezl $1, global_label         # encoding: [0x54,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bltul $7,$0,local_label
# CHECK: bnez $7, local_label           # encoding: [0x14,0xe0,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bltul $0,$8,local_label
# CHECK: bnez $8, local_label           # encoding: [0x15,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bltul $0,$0,local_label
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]

  bleul $7,$8,local_label
# CHECK: sltu $1, $8, $7                # encoding: [0x01,0x07,0x08,0x2b]
# CHECK: beqzl $1, local_label          # encoding: [0x50,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bleul $7,$8,global_label
# CHECK: sltu $1, $8, $7                # encoding: [0x01,0x07,0x08,0x2b]
# CHECK: beqzl $1, global_label         # encoding: [0x50,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bleul $7,$0,local_label
# CHECK: beqz $7, local_label           # encoding: [0x10,0xe0,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bleul $0,$8,local_label
# CHECK: beqz $8, local_label           # encoding: [0x11,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bleul $0,$0,local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: b local_label                  # encoding: [0x10,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]

  bgeul $7,$8,local_label
# CHECK: sltu $1, $7, $8                # encoding: [0x00,0xe8,0x08,0x2b]
# CHECK: beqzl $1, local_label          # encoding: [0x50,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgeul $7,$8,global_label
# CHECK: sltu $1, $7, $8                # encoding: [0x00,0xe8,0x08,0x2b]
# CHECK: beqzl $1, global_label         # encoding: [0x50,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgeul $7,$0,local_label
# CHECK: beqz $7, local_label           # encoding: [0x10,0xe0,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgeul $0,$8,local_label
# CHECK: beqz $8, local_label           # encoding: [0x11,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgeul $0,$0,local_label
# WARNING: :[[@LINE-1]]:3: warning: branch is always taken
# CHECK: b local_label                  # encoding: [0x10,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]

  bgtul $7,$8,local_label
# CHECK: sltu $1, $8, $7                # encoding: [0x01,0x07,0x08,0x2b]
# CHECK: bnezl $1, local_label          # encoding: [0x54,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgtul $7,$8,global_label
# CHECK: sltu $1, $8, $7                # encoding: [0x01,0x07,0x08,0x2b]
# CHECK: bnezl $1, global_label         # encoding: [0x54,0x20,A,A]
# CHECK:                                #   fixup A - offset: 0, value: global_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgtul $7,$0,local_label
# CHECK: bnez $7, local_label           # encoding: [0x14,0xe0,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgtul $0,$8,local_label
# CHECK: bnez $8, local_label           # encoding: [0x15,0x00,A,A]
# CHECK:                                #   fixup A - offset: 0, value: local_label-4, kind: fixup_Mips_PC16
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
  bgtul $0,$0,local_label
# CHECK: nop                            # encoding: [0x00,0x00,0x00,0x00]
