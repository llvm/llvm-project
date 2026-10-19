# RUN: llvm-mc -triple=agx -mcpu=agx-g13g -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=agx -mcpu=agx-g14c -show-encoding %s | FileCheck %s

# Encodings cross-checked against the applegpu reference assembler and
# disassembler (dougallj/applegpu): each case is a random canonical encoding
# that applegpu disassembles to the same text.

else_fcmp r0l, gte, u217.neg, r101l.cache, 1
# CHECK: else_fcmp r0l, gte, u217.neg, r101l.cache, 1{{ +}}// encoding: [0x42,0xaa,0xf2,0xa9,0x08,0x0b]
else_fcmp r0l, ngte, u196l.abs, r60.discard.abs, 1
# CHECK: else_fcmp r0l, ngte, u196l.abs, r60.discard.abs, 1{{ +}}// encoding: [0x42,0xab,0x48,0x85,0x6f,0x09]
else_fcmp r0l.cache, lte, r108.cache.abs, u221h, 2
# CHECK: else_fcmp r0l.cache, lte, r108.cache.abs, u221h, 2{{ +}}// encoding: [0xc2,0xd2,0x98,0xb6,0x17,0x0e]
else_fcmp r0l.cache, nltn, r10.abs, r110l.neg, 2
# CHECK: else_fcmp r0l.cache, nltn, r10.abs, r110l.neg, 2{{ +}}// encoding: [0xc2,0x73,0x54,0xc6,0x85,0x03]

else_icmp r0l, ult, u197l, r65l, 1
# CHECK: else_icmp r0l, ult, u197l, r65l, 1{{ +}}// encoding: [0x52,0x2a,0x4a,0x21,0x04,0x0a]
else_icmp r0l, ueq, u106h, 2, 0
# CHECK: else_icmp r0l, ueq, u106h, 2, 0{{ +}}// encoding: [0x52,0x02,0x15,0x21,0x00,0x0c]
else_icmp r0l, seq, u161h, r74h, 0
# CHECK: else_icmp r0l, seq, u161h, r74h, 0{{ +}}// encoding: [0x52,0x82,0x43,0x51,0x05,0x06]
else_icmp r0l, ult, r81l.discard, u17, 1
# CHECK: else_icmp r0l, ult, r81l.discard, u17, 1{{ +}}// encoding: [0x52,0x2a,0xe2,0x20,0x1a,0x08]

if_fcmp r0l, lt, u179h.abs, r104h, 0
# CHECK: if_fcmp r0l, lt, u179h.abs, r104h, 0{{ +}}// encoding: [0x42,0x20,0x67,0x15,0x05,0x07]
if_fcmp r0l.cache, ngtn, u200.neg, u15l.neg, 2
# CHECK: if_fcmp r0l.cache, ngtn, u200.neg, u15l.neg, 2{{ +}}// encoding: [0xc2,0xf1,0xd0,0xe9,0x91,0x08]
if_fcmp r0l.cache, lte, r122h.discard, r100l.neg, 0
# CHECK: if_fcmp r0l.cache, lte, r122h.discard, r100l.neg, 0{{ +}}// encoding: [0xc2,0xc0,0xf5,0x80,0x84,0x0f]
if_fcmp r0l, nlte, u145, r38.cache, 3
# CHECK: if_fcmp r0l, nlte, u145, r38.cache, 3{{ +}}// encoding: [0x42,0xd9,0xe2,0xc1,0x28,0x01]

if_icmp r0l, ugte, u24l, u47h, 0
# CHECK: if_icmp r0l, ugte, u24l, u47h, 0{{ +}}// encoding: [0x52,0x21,0x30,0xf1,0x11,0x01]
if_icmp r0l, slte, u29l, u200, 0
# CHECK: if_icmp r0l, slte, u29l, u200, 0{{ +}}// encoding: [0x52,0xc1,0x3a,0x01,0x1d,0x02]
if_icmp r0l, ugt, u92h, r108l, 2
# CHECK: if_icmp r0l, ugt, u92h, r108l, 2{{ +}}// encoding: [0x52,0x50,0x39,0x81,0x05,0x0b]
if_icmp r0l.cache, seq, u81l, r114h, 0
# CHECK: if_icmp r0l.cache, seq, u81l, r114h, 0{{ +}}// encoding: [0xd2,0x80,0x22,0x51,0x06,0x0b]



memory_barrier 1, 0, 15
# CHECK: memory_barrier 1, 0, 15{{ +}}// encoding: [0xf5,0xf4]
memory_barrier 3, 3, 13
# CHECK: memory_barrier 3, 3, 13{{ +}}// encoding: [0xf5,0xdf]
memory_barrier 3, 0, 9
# CHECK: memory_barrier 3, 0, 9{{ +}}// encoding: [0xf5,0x9c]
memory_barrier 2, 1, 12
# CHECK: memory_barrier 2, 1, 12{{ +}}// encoding: [0xf5,0xc9]

pop_exec r0l, 0
# CHECK: pop_exec r0l, 0{{ +}}// encoding: [0x52,0x06,0x00,0x00,0x00,0x00]
pop_exec r0l, 3
# CHECK: pop_exec r0l, 3{{ +}}// encoding: [0x52,0x1e,0x00,0x00,0x00,0x00]
pop_exec r0l.cache, 3
# CHECK: pop_exec r0l.cache, 3{{ +}}// encoding: [0xd2,0x1e,0x00,0x00,0x00,0x00]
pop_exec r0l.cache, 2
# CHECK: pop_exec r0l.cache, 2{{ +}}// encoding: [0xd2,0x16,0x00,0x00,0x00,0x00]

stop
# CHECK: stop{{ +}}// encoding: [0x88,0x00]
stop
# CHECK: stop{{ +}}// encoding: [0x88,0x00]
stop
# CHECK: stop{{ +}}// encoding: [0x88,0x00]
stop
# CHECK: stop{{ +}}// encoding: [0x88,0x00]

threadgroup_barrier
# CHECK: threadgroup_barrier{{ +}}// encoding: [0x68,0x00]
threadgroup_barrier
# CHECK: threadgroup_barrier{{ +}}// encoding: [0x68,0x00]
threadgroup_barrier
# CHECK: threadgroup_barrier{{ +}}// encoding: [0x68,0x00]
threadgroup_barrier
# CHECK: threadgroup_barrier{{ +}}// encoding: [0x68,0x00]

trap
# CHECK: trap{{ +}}// encoding: [0x08,0x00]
trap
# CHECK: trap{{ +}}// encoding: [0x08,0x00]
trap
# CHECK: trap{{ +}}// encoding: [0x08,0x00]
trap
# CHECK: trap{{ +}}// encoding: [0x08,0x00]

wait 1
# CHECK: wait 1{{ +}}// encoding: [0x38,0x01]
wait 0
# CHECK: wait 0{{ +}}// encoding: [0x38,0x00]
wait 0
# CHECK: wait 0{{ +}}// encoding: [0x38,0x00]
wait 1
# CHECK: wait 1{{ +}}// encoding: [0x38,0x01]

while_fcmp r0l, gtn, r77l.discard, u38.abs.neg, 2
# CHECK: while_fcmp r0l, gtn, r77l.discard, u38.abs.neg, 2{{ +}}// encoding: [0x42,0xf4,0xda,0xc0,0xd8,0x09]
while_fcmp r0l.cache, ltn, r35l.abs.neg, 3.125, 1
# CHECK: while_fcmp r0l.cache, ltn, r35l.abs.neg, 3.125, 1{{ +}}// encoding: [0xc2,0x6c,0x46,0x9c,0x00,0x05]
while_fcmp r0l, lte, r10.neg, u167l.abs.neg, 2
# CHECK: while_fcmp r0l, lte, r10.neg, u167l.abs.neg, 2{{ +}}// encoding: [0x42,0xd4,0x54,0xea,0xd4,0x01]
while_fcmp r0l, lt, r81h.cache.neg, u131, 3
# CHECK: while_fcmp r0l, lt, r81h.cache.neg, u131, 3{{ +}}// encoding: [0x42,0x3c,0xa3,0x68,0x1c,0x08]

while_icmp r0l, ueq, r83, u5, 2
# CHECK: while_icmp r0l, ueq, r83, u5, 2{{ +}}// encoding: [0x52,0x14,0x66,0xa2,0x18,0x08]
while_icmp r0l.cache, ulte, r48h.cache, u64, 1
# CHECK: while_icmp r0l.cache, ulte, r48h.cache, u64, 1{{ +}}// encoding: [0xd2,0x4d,0xa1,0x00,0x18,0x06]
while_icmp r0l.cache, seq, r20h.discard, u43l, 3
# CHECK: while_icmp r0l.cache, seq, r20h.discard, u43l, 3{{ +}}// encoding: [0xd2,0x9c,0xe9,0x60,0x11,0x01]
while_icmp r0l.cache, seq, r45l, r73l, 3
# CHECK: while_icmp r0l.cache, seq, r45l, r73l, 3{{ +}}// encoding: [0xd2,0x9c,0x5a,0x20,0x05,0x06]

