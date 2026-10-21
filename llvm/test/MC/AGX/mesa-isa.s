# Instructions Mesa's AGX compiler emits are accepted in Mesa ISA mode, and
# every other one is an error naming the closest Mesa form.
# RUN: not llvm-mc -triple=agx -mcpu=agx-g14c -mattr=+mesa-isa -show-encoding %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
# RUN: not llvm-mc -triple=agx -mcpu=agx-g14d -mattr=+mesa-isa -show-encoding %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=G14D
# RUN: llvm-mc -triple=agx -mcpu=agx-g14c -show-encoding %s -o /dev/null

# Mesa forms.
wait 0
stop
threadgroup_barrier
device_load 0, i32, x, r1, u0_u1, 0, signed
atomic cmpxchg, 0, r1, u0_u1, 0, signed, r2_r3, 0

# Bit 45 of a device atomic belongs to multi-cluster G13 and two-die parts.
atomic cmpxchg, 0, r1, u0_u1, 0, signed, r2_r3, 1
# CHECK: :[[@LINE-1]]:1: error: ATOMIC_g13 is not an instruction Mesa emits (mesa-isa): atomic.coherent: not emitted for this GPU
# G14D-NOT: :[[@LINE-2]]:1: error

# The G14 float add (sub with bit 45) is never emitted by Mesa.
atomic sub, 0, r1, u0_u1, 0, signed, r1, 1
# CHECK: :[[@LINE-1]]:1: error: ATOMIC_g13 is not an instruction Mesa emits (mesa-isa): atomic.coherent: field value
# G14D: :[[@LINE-2]]:1: error: ATOMIC_g13 is not an instruction Mesa emits (mesa-isa): atomic.coherent: field value

# The G14 block barrier is not one of Mesa's instructions.
block_barrier
# CHECK: :[[@LINE-1]]:1: error: BLOCK_BARRIER_g13 is not an instruction Mesa emits (mesa-isa): no Mesa instruction has this opcode and length
# G14D: :[[@LINE-2]]:1: error: BLOCK_BARRIER_g13 is not an instruction Mesa emits (mesa-isa): no Mesa instruction has this opcode and length

# shlhi is decoded by Mesa's disassembler but never emitted by its compiler.
shlhi r118l, r109.discard, r119l.cache, r9, mask 0xFF
# CHECK: :[[@LINE-1]]:1: error: SHLHI_g13 is not an instruction Mesa emits (mesa-isa): no Mesa instruction has this opcode and length
