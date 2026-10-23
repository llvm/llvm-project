# RUN: llvm-mc -triple=agx -mcpu=agx-g14c -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=agx -mcpu=agx-g14c -filetype=obj %s -o - | llvm-objdump -d - | FileCheck %s --check-prefix=DIS

# These forms match the G14C call/return probes. LLVM code generation does
# not yet use them; the calling convention and per-call stack are separate.
call callee
# ENC: call callee
# DIS: 10 c0 0e 00 00 00{{ *}}call 0xe <callee>
pop_exec r0l, 2
stop
callee:
ret r1
# ENC: ret r1{{ *}}// encoding: [0x14,0x02]
# DIS: 14 02{{ *}}ret r1
call_reg r30
# ENC: call_reg r30{{ *}}// encoding: [0x04,0x3c]
# DIS: 04 3c{{ *}}call_reg r30
