# RUN: not llvm-mc -triple=csky %s -filetype=obj -o %t.o 2>&1 | FileCheck %s

# Out of PC range

# br/bt/bf

.L.test1:
.space 0x10001
br32 .L.test1 # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value
              # CHECK: :[[@LINE-1]]:1: error: fixup value must be 2-byte aligned

br32 .L.test2 # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value
              # CHECK: :[[@LINE-1]]:1: error: fixup value must be 2-byte aligned
.space 0x10001
.L.test2:

.L.test3:
.space 0xFFFF
br32 .L.test3 # CHECK: :[[@LINE]]:1: error: fixup value must be 2-byte aligned

.L.test4:
.space 0x10002
br32 .L.test4 # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value

# bsr
.L.test5:
.space 0x4000001
bsr32 .L.test5 # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value
               # CHECK: :[[@LINE-1]]:1: error: fixup value must be 2-byte aligned

bsr32 .L.test6 # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value
               # CHECK: :[[@LINE-1]]:1: error: fixup value must be 2-byte aligned
.space 0x4000001
.L.test6:

.L.test7:
.space 0x3FFFFFF
bsr32 .L.test7 # CHECK: :[[@LINE]]:1: error: fixup value must be 2-byte aligned

.L.test8:
.space 0x4000002
bsr32 .L.test8 # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value

# grs
.L.test9:
.space 0x40001
grs32 a0, .L.test9 # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value
                   # CHECK: :[[@LINE-1]]:1: error: fixup value must be 2-byte aligned

grs32 a0, .L.test10 # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value
                    # CHECK: :[[@LINE-1]]:1: error: fixup value must be 2-byte aligned
.space 0x40001
.L.test10:

.L.test11:
.space 0x3FFFF
grs32 a0, .L.test11 # CHECK: :[[@LINE]]:1: error: fixup value must be 2-byte aligned

.L.test12:
.space 0x40002
grs32 a0, .L.test12 # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value


# TODO: Fixup
lrw32 a0, [.L.test15] # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value
                    # CHECK: :[[@LINE-1]]:1: error: fixup value must be 4-byte aligned
.space 0x40001
.L.test15:

# TODO: Fixup
jsri32 [.L.test16]     # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value
                     # CHECK: :[[@LINE-1]]:1: error: fixup value must be 4-byte aligned
.space 0x40001
.L.test16:

# TODO: Fixup
jmpi32 [.L.test17]     # CHECK: :[[@LINE]]:1: error: out of range pc-relative fixup value
                     # CHECK: :[[@LINE-1]]:1: error: fixup value must be 4-byte aligned
.space 0x40001
.L.test17:
