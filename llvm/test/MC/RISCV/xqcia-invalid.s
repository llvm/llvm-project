# Xqcia - Qualcomm uC Arithmetic Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcia < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcia < %s

qc.shlsat x10, x3, 17
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlsat x10, x3
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlsat x0, x3, x17
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlsat x10, x0, x17
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlsat x10, x3, x0
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlsat x10, x3, x17 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.shlusat x23, x25, 27
# plus-error@-1:22 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlusat x23, x25
# plus-error@-1:20 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlusat x0, x25, x27
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlusat x23, x0, x27
# plus-error@-1:17 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlusat x23, x25, x0
# plus-error@-1:22 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.shlusat x23, x25, x27 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.addsat x17, x14, 7
# plus-error@-1:21 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.addsat x17, x14
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.addsat x0, x14, x7
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.addsat x17, x0, x7
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.addsat x17, x14, x0
# plus-error@-1:21 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.addsat x17, x14, x7 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.addusat x8, x18, 28
# plus-error@-1:21 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.addusat x8, x18
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.addusat x0, x18, x28
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.addusat x8, x0, x28
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.addusat x8, x18, x0
# plus-error@-1:21 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.addusat x8, x18, x28 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.subsat x22, x2, 12
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.subsat x22, x2
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.subsat x0, x2, x12
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.subsat x22, x0, x12
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.subsat x22, x2, x0
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.subsat x22, x2, x12 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.subusat x9, x14, 17
# plus-error@-1:21 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.subusat x9, x14
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.subusat x0, x14, x17
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.subusat x9, x0, x17
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.subusat x9, x14, x0
# plus-error@-1:21 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.subusat x9, x14, x17 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.wrap x3, x30, 23
# plus-error@-1:18 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.wrap x3, x30
# plus-error@-1:16 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.wrap x0, x30, x23
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.wrap x3, x30, x0
# plus-error@-1:18 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.wrap x3, x30, x23 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.wrapi x0, 12, 2047
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}

qc.wrapi x0, x12, 2047
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.wrapi x6, x0, 2047
# plus-error@-1:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.wrapi x6, x12
# plus-error@-1:17 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.wrapi x6, x12, 2048
# plus-error@-1:19 {{immediate must be an integer in the range [0, 2047]}}
# minus-error@-2 {{invalid instruction}}

qc.wrapi x6, x12, 2047 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.norm x3, 7
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.norm x3
# plus-error@-1:11 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.norm x0, x7
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.norm x3, x0
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.norm x3, x7 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.normu x11, 17
# plus-error@-1:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.normu x11
# plus-error@-1:13 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.normu x0, x17
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.normu x11, x0
# plus-error@-1:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.normu x11, x17 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}


qc.normeu x26, 31
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.normeu x26
# plus-error@-1:14 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.normeu x0, x31
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.normeu x26, x0
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.normeu x26, x31 # minus-error:1 {{instruction requires the following: 'Xqcia' (Qualcomm uC Arithmetic Extension)}}
