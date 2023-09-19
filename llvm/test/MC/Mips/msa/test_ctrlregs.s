# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
#CHECK:  cfcmsa       $1, $0                  # encoding: [0x78,0x7e,0x00,0x59]
#CHECK:  cfcmsa       $1, $0                  # encoding: [0x78,0x7e,0x00,0x59]
#CHECK:  cfcmsa       $2, $1                  # encoding: [0x78,0x7e,0x08,0x99]
#CHECK:  cfcmsa       $2, $1                  # encoding: [0x78,0x7e,0x08,0x99]
#CHECK:  cfcmsa       $3, $2                  # encoding: [0x78,0x7e,0x10,0xd9]
#CHECK:  cfcmsa       $3, $2                  # encoding: [0x78,0x7e,0x10,0xd9]
#CHECK:  cfcmsa       $4, $3                  # encoding: [0x78,0x7e,0x19,0x19]
#CHECK:  cfcmsa       $4, $3                  # encoding: [0x78,0x7e,0x19,0x19]
#CHECK:  cfcmsa       $5, $4                  # encoding: [0x78,0x7e,0x21,0x59]
#CHECK:  cfcmsa       $5, $4                  # encoding: [0x78,0x7e,0x21,0x59]
#CHECK:  cfcmsa       $6, $5                  # encoding: [0x78,0x7e,0x29,0x99]
#CHECK:  cfcmsa       $6, $5                  # encoding: [0x78,0x7e,0x29,0x99]
#CHECK:  cfcmsa       $7, $6                  # encoding: [0x78,0x7e,0x31,0xd9]
#CHECK:  cfcmsa       $7, $6                  # encoding: [0x78,0x7e,0x31,0xd9]
#CHECK:  cfcmsa       $8, $7                  # encoding: [0x78,0x7e,0x3a,0x19]
#CHECK:  cfcmsa       $8, $7                  # encoding: [0x78,0x7e,0x3a,0x19]

#CHECK:  ctcmsa       $0, $1                  # encoding: [0x78,0x3e,0x08,0x19]
#CHECK:  ctcmsa       $0, $1                  # encoding: [0x78,0x3e,0x08,0x19]
#CHECK:  ctcmsa       $1, $2                  # encoding: [0x78,0x3e,0x10,0x59]
#CHECK:  ctcmsa       $1, $2                  # encoding: [0x78,0x3e,0x10,0x59]
#CHECK:  ctcmsa       $2, $3                  # encoding: [0x78,0x3e,0x18,0x99]
#CHECK:  ctcmsa       $2, $3                  # encoding: [0x78,0x3e,0x18,0x99]
#CHECK:  ctcmsa       $3, $4                  # encoding: [0x78,0x3e,0x20,0xd9]
#CHECK:  ctcmsa       $3, $4                  # encoding: [0x78,0x3e,0x20,0xd9]
#CHECK:  ctcmsa       $4, $5                  # encoding: [0x78,0x3e,0x29,0x19]
#CHECK:  ctcmsa       $4, $5                  # encoding: [0x78,0x3e,0x29,0x19]
#CHECK:  ctcmsa       $5, $6                  # encoding: [0x78,0x3e,0x31,0x59]
#CHECK:  ctcmsa       $5, $6                  # encoding: [0x78,0x3e,0x31,0x59]
#CHECK:  ctcmsa       $6, $7                  # encoding: [0x78,0x3e,0x39,0x99]
#CHECK:  ctcmsa       $6, $7                  # encoding: [0x78,0x3e,0x39,0x99]
#CHECK:  ctcmsa       $7, $8                  # encoding: [0x78,0x3e,0x41,0xd9]
#CHECK:  ctcmsa       $7, $8                  # encoding: [0x78,0x3e,0x41,0xd9]

cfcmsa       $1, $msair
cfcmsa       $1, $0
cfcmsa       $2, $msacsr
cfcmsa       $2, $1
cfcmsa       $3, $msaaccess
cfcmsa       $3, $2
cfcmsa       $4, $msasave
cfcmsa       $4, $3
cfcmsa       $5, $msamodify
cfcmsa       $5, $4
cfcmsa       $6, $msarequest
cfcmsa       $6, $5
cfcmsa       $7, $msamap
cfcmsa       $7, $6
cfcmsa       $8, $msaunmap
cfcmsa       $8, $7

ctcmsa       $msair, $1
ctcmsa       $0, $1
ctcmsa       $msacsr, $2
ctcmsa       $1, $2
ctcmsa       $msaaccess, $3
ctcmsa       $2, $3
ctcmsa       $msasave, $4
ctcmsa       $3, $4
ctcmsa       $msamodify, $5
ctcmsa       $4, $5
ctcmsa       $msarequest, $6
ctcmsa       $5, $6
ctcmsa       $msamap, $7
ctcmsa       $6, $7
ctcmsa       $msaunmap, $8
ctcmsa       $7, $8
