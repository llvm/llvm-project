# Xqcibi - Qualcomm uC Branch Immediate Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcibi < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcibi < %s

qc.beqi x0, 12, 346
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.beqi x8, 12
# plus-error@-1:15 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.beqi x8, 22, 346
# plus-error@-1:13 {{immediate must be non-zero in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.beqi x8, 12, 1211
# plus-error@-1:17 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.beqi x8, 12, 346 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.bnei x0, 15, 4094
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.bnei x4, 15
# plus-error@-1:15 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.bnei x4, -45, 4094
# plus-error@-1:13 {{immediate must be non-zero in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.bnei x4, 15, 5000
# plus-error@-1:17 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.bnei x4, 15, 4094 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.bgei x0, 1, -4096
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.bgei x10, 1
# plus-error@-1:15 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.bgei x10, 21, -4096
# plus-error@-1:14 {{immediate must be non-zero in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.bgei x10, 1, -4098
# plus-error@-1:17 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.bgei x10, 1, -4096 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.blti x0, 6, 2000
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.blti x1, 6
# plus-error@-1:14 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.blti x1, 56, 2000
# plus-error@-1:13 {{immediate must be non-zero in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.blti x1, 6, 12000
# plus-error@-1:16 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.blti x1, 6, 2000 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.bgeui x0, 11, 128
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.bgeui x12, 11
# plus-error@-1:17 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.bgeui x12, 41, 128
# plus-error@-1:15 {{immediate must be an integer in the range [1, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.bgeui x12, 11, 11128
# plus-error@-1:19 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.bgeui x12, 11, 128 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.bltui x0, 7, 666
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.bltui x2, 7
# plus-error@-1:15 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.bltui x2, -7, 666
# plus-error@-1:14 {{immediate must be an integer in the range [1, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.bltui x2, 7, -6666
# plus-error@-1:17 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.bltui x2, 7, 666 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.e.beqi x0, 1, 2
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.beqi x1, 1
# plus-error@-1:16 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.beqi x1, 32768, 2
# plus-error@-1:15 {{immediate must be non-zero in the range [-32768, 32767]}}
# minus-error@-2 {{invalid instruction}}

qc.e.beqi x1, 1, 21
# plus-error@-1:18 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.e.beqi x1, 1, 2 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.e.bnei x0, 115, 4094
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.bnei x4, 115
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.bnei x4, -33115, 4094
# plus-error@-1:15 {{immediate must be non-zero in the range [-32768, 32767]}}
# minus-error@-2 {{invalid instruction}}

qc.e.bnei x4, 115, 211
# plus-error@-1:20 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.e.bnei x4, 115, 4094 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.e.bgei x0, -32768, -4096
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.bgei x10, -32768
# plus-error@-1:22 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.bgei x10, -32769, -4096
# plus-error@-1:16 {{immediate must be non-zero in the range [-32768, 32767]}}
# minus-error@-2 {{invalid instruction}}

qc.e.bgei x10, -32768, -4097
# plus-error@-1:24 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.e.bgei x10, -32768, -4096 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.e.blti x0, 32767, 2000
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.blti x1, 32767
# plus-error@-1:20 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.blti x1, 42767, 2000
# plus-error@-1:15 {{immediate must be non-zero in the range [-32768, 32767]}}
# minus-error@-2 {{invalid instruction}}

qc.e.blti x1, 32767, 2001
# plus-error@-1:22 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.e.blti x1, 32767, 2000 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.e.bgeui x0, 711, 128
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.bgeui x12, 711
# plus-error@-1:20 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.bgeui x12, 0, 128
# plus-error@-1:17 {{immediate must be an integer in the range [1, 65535]}}
# minus-error@-2 {{invalid instruction}}

qc.e.bgeui x12, 711, 129
# plus-error@-1:22 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.e.bgeui x12, 711, 128 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}


qc.e.bltui x0, 7, 666
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.bltui x2, 7
# plus-error@-1:17 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.bltui x2, -7, 666
# plus-error@-1:16 {{immediate must be an integer in the range [1, 65535]}}
# minus-error@-2 {{invalid instruction}}

qc.e.bltui x2, 7, 667
# plus-error@-1:19 {{immediate must be a multiple of 2 bytes in the range [-4096, 4094]}}
# minus-error@-2 {{invalid instruction}}

qc.e.bltui x2, 7, 666 # minus-error:1 {{instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)}}
