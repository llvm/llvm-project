# REQUIRES: aarch64-registered-target, exegesis-can-measure-latency

# LLVM-EXEGESIS-MEM-DEF test_mem 4096 16
# LLVM-EXEGESIS-MEM-MAP test_mem 140737488093184
# LLVM-EXEGESIS-DEFREG X0 65536
# LLVM-EXEGESIS-DEFREG X1 0
.arch armv8-a+sve

# memory location = VAddressSpaceCeiling - Pagesize * var
# Aux memory loc  =     0x0x800000000000 -  0x10000 * 2  = 0x7ffffffe0000      
mov x0, 140737488224256
ldr x1, [x0, #0]

# specific mem loc =    0x0x800000000000 -  0x10000 * 4  = 0x7ffffffc0000
mov x0, 140737488093184
ldr x1, [x0, #0]


# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --execution-mode=subprocess \
# RUN:               --mode=inverse_throughput --benchmark-phase=assemble-measured-code \
# RUN:               --dump-object-to-disk=%t.o --min-instructions=1 --snippets-file=%s 2>&1

# RUN: llvm-objdump -d %t.o > %t.disasm
# RUN: FileCheck %s --check-prefix=CHECK_SYSCALLS < %t.disasm

# CHECK_SYSCALLS: <foo>:

# Check for aux memory mapping syscall (syscall number 222/0xde)
# CHECK_SYSCALLS: mov     x0, #0x7ffffffe0000
# CHECK_SYSCALLS-NEXT: mov     x1, #0x1000
# CHECK_SYSCALLS-NEXT: mov     x2, #0x3
# CHECK_SYSCALLS-NEXT: mov     x3, #0x21
# CHECK_SYSCALLS-NEXT: movk    x3, #0x10, lsl #16
# CHECK_SYSCALLS-NEXT: mov     x4, #-0x1
# CHECK_SYSCALLS-NEXT: mov     x5, #0x0
# CHECK_SYSCALLS-NEXT: mov     x8, #0xde
# CHECK_SYSCALLS-NEXT: svc     #0

# CHECK_SYSCALLS: str     x0, [sp, #-0x10]!

# Check for specific memory mapping syscall
# CHECK_SYSCALLS: mov     x0, #0x7ffffffc0000
# CHECK_SYSCALLS-NEXT: mov     x1, #0x1000
# CHECK_SYSCALLS-NEXT: mov     x2, #0x3
# CHECK_SYSCALLS-NEXT: mov     x3, #0x21
# CHECK_SYSCALLS-NEXT: movk    x3, #0x10, lsl #16
# CHECK_SYSCALLS-NEXT: mov     x4, #-0x1
# CHECK_SYSCALLS-NEXT: mov     x5, #0x0
# CHECK_SYSCALLS-NEXT: mov     x8, #0xde
# CHECK_SYSCALLS-NEXT: svc     #0

# CHECK_SYSCALLS: ldr     x0, [sp], #0x10
# CHECK_SYSCALLS: mov     x1, #0x0

# Check for performance counter control syscalls (ioctl - syscall number 29/0x1d)
# CHECK_SYSCALLS: str     x8, [sp, #-0x10]!
# CHECK_SYSCALLS-NEXT: str     x0, [sp, #-0x10]!
# CHECK_SYSCALLS-NEXT: str     x1, [sp, #-0x10]!
# CHECK_SYSCALLS-NEXT: str     x2, [sp, #-0x10]!
# CHECK_SYSCALLS-NEXT: mov     x16, #0x7ffffffe0000
# CHECK_SYSCALLS-NEXT: ldr     w0, [x16]
# CHECK_SYSCALLS-NEXT: mov     x1, #0x2403
# CHECK_SYSCALLS-NEXT: mov     x2, #0x1
# CHECK_SYSCALLS-NEXT: mov     x8, #0x1d
# CHECK_SYSCALLS-NEXT: svc     #0
# CHECK_SYSCALLS-NEXT: ldr     x2, [sp], #0x10
# CHECK_SYSCALLS-NEXT: ldr     x1, [sp], #0x10
# CHECK_SYSCALLS-NEXT: ldr     x0, [sp], #0x10
# CHECK_SYSCALLS-NEXT: ldr     x8, [sp], #0x10

# === Test instruction execution ===
# CHECK_SYSCALLS: mov     x0, #0x7ffffffe0000
# CHECK_SYSCALLS-NEXT: ldr     x1, [x0]
# CHECK_SYSCALLS-NEXT: mov     x0, #0x7ffffffc0000
# CHECK_SYSCALLS-NEXT: ldr     x1, [x0]

# === ioctl syscall - stop performance counters ===
# CHECK_SYSCALLS: mov     x16, #0x7ffffffe0000
# CHECK_SYSCALLS-NEXT: ldr     w0, [x16]
# CHECK_SYSCALLS-NEXT: mov     x1, #0x2401
# CHECK_SYSCALLS-NEXT: mov     x2, #0x1
# CHECK_SYSCALLS-NEXT: mov     x8, #0x1d
# CHECK_SYSCALLS-NEXT: svc     #0

# Check for process exit syscall (exit - syscall number 93/0x5d)  
# CHECK_SYSCALLS: mov     x0, #0x0
# CHECK_SYSCALLS-NEXT: mov     x8, #0x5d
# CHECK_SYSCALLS-NEXT: svc     #0

# CHECK_SYSCALLS-NEXT: ret
