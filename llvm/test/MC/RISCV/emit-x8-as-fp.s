# RUN: llvm-mc --triple=riscv32 --show-encoding < %s 2>&1 \
# RUN:   | FileCheck --check-prefix=DEFAULT %s
# RUN: llvm-mc --triple=riscv64 --show-encoding < %s 2>&1 \
# RUN:   | FileCheck --check-prefix=DEFAULT %s
# RUN: llvm-mc --triple=riscv32 -M emit-x8-as-fp --show-encoding < %s 2>&1 \
# RUN:   | FileCheck --check-prefix=EMIT-FP %s
# RUN: llvm-mc --triple=riscv64 -M emit-x8-as-fp --show-encoding < %s 2>&1 \
# RUN:   | FileCheck --check-prefix=EMIT-FP %s
# RUN: llvm-mc --triple=riscv32 -M numeric -M emit-x8-as-fp --show-encoding \
# RUN:   < %s 2>&1 | FileCheck --check-prefix=NUMERIC %s
# RUN: llvm-mc --triple=riscv64 -M numeric -M emit-x8-as-fp --show-encoding \
# RUN:   < %s 2>&1 | FileCheck --check-prefix=NUMERIC %s

# DEFAULT: sw      a0, -12(s0)                     # encoding: [0x23,0x2a,0xa4,0xfe]
# EMIT-FP: sw      a0, -12(fp)                     # encoding: [0x23,0x2a,0xa4,0xfe]
# NUMERIC: sw      x10, -12(x8)                    # encoding: [0x23,0x2a,0xa4,0xfe]
sw a0, -12(s0)

# DEFAULT: lw      a0, -12(s0)                     # encoding: [0x03,0x25,0x44,0xff]
# EMIT-FP: lw      a0, -12(fp)                     # encoding: [0x03,0x25,0x44,0xff]
# NUMERIC: lw      x10, -12(x8)                    # encoding: [0x03,0x25,0x44,0xff]
lw a0, -12(s0)

# DEFAULT: sw      a0, -12(s0)                     # encoding: [0x23,0x2a,0xa4,0xfe]
# EMIT-FP: sw      a0, -12(fp)                     # encoding: [0x23,0x2a,0xa4,0xfe]
# NUMERIC: sw      x10, -12(x8)                    # encoding: [0x23,0x2a,0xa4,0xfe]
sw a0, -12(fp)

# DEFAULT: lw      a0, -12(s0)                     # encoding: [0x03,0x25,0x44,0xff]
# EMIT-FP: lw      a0, -12(fp)                     # encoding: [0x03,0x25,0x44,0xff]
# NUMERIC: lw      x10, -12(x8)                    # encoding: [0x03,0x25,0x44,0xff]
lw a0, -12(fp)

# DEFAULT: sw      a0, -12(s0)                     # encoding: [0x23,0x2a,0xa4,0xfe]
# EMIT-FP: sw      a0, -12(fp)                     # encoding: [0x23,0x2a,0xa4,0xfe]
# NUMERIC: sw      x10, -12(x8)                    # encoding: [0x23,0x2a,0xa4,0xfe]
sw a0, -12(x8)

# DEFAULT: lw      a0, -12(s0)                     # encoding: [0x03,0x25,0x44,0xff]
# EMIT-FP: lw      a0, -12(fp)                     # encoding: [0x03,0x25,0x44,0xff]
# NUMERIC: lw      x10, -12(x8)                    # encoding: [0x03,0x25,0x44,0xff]
lw a0, -12(x8)
