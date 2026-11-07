# Xqcilo - Qualcomm uC Large Offset Load Store extension
# RUN: llvm-mc --verify=enabled %s -triple=riscv32 -mattr=+xqcilo,+xqcili
# RUN: llvm-mc --verify=disabled %s -triple=riscv32 -mattr=-xqcilo

# enabled-error@+4:1 {{invalid instruction, any one of the following would fix this:}}
# enabled-note@+3:13 {{operand must be a bare symbol name}}
# enabled-note@+2:19 {{too few operands for instruction}}
# disabled-error@+1:1 {{invalid instruction}}
qc.e.lb a0, 0xf000

# enabled-error@+4:1 {{invalid instruction, any one of the following would fix this:}}
# enabled-note@+3:14 {{operand must be a bare symbol name}}
# enabled-note@+2:20 {{too few operands for instruction}}
# disabled-error@+1:1 {{invalid instruction}}
qc.e.lbu a0, 0xf000

# enabled-error@+4:1 {{invalid instruction, any one of the following would fix this:}}
# enabled-note@+3:13 {{operand must be a bare symbol name}}
# enabled-note@+2:19 {{too few operands for instruction}}
# disabled-error@+1:1 {{invalid instruction}}
qc.e.lh a0, 0xf000

# enabled-error@+4:1 {{invalid instruction, any one of the following would fix this:}}
# enabled-note@+3:14 {{operand must be a bare symbol name}}
# enabled-note@+2:20 {{too few operands for instruction}}
# disabled-error@+1:1 {{invalid instruction}}
qc.e.lhu a0, 0xf000

# enabled-error@+4:1 {{invalid instruction, any one of the following would fix this:}}
# enabled-note@+3:13 {{operand must be a bare symbol name}}
# enabled-note@+2:19 {{too few operands for instruction}}
# disabled-error@+1:1 {{invalid instruction}}
qc.e.lw a0, 0xf000

# enabled-error@+2:13 {{operand must be a bare symbol name}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sb a0, 0xf000, t0

# enabled-error@+2:13 {{operand must be a bare symbol name}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sh a0, 0xf000, t0

# enabled-error@+2:13 {{operand must be a bare symbol name}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sw a0, 0xf000, t0

# disabled-error@+1:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lb a0, undefined
# disabled-error@+1:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lbu a0, undefined
# disabled-error@+1:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lh a0, undefined
# disabled-error@+1:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lhu a0, undefined
# disabled-error@+1:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lw a0, undefined
# disabled-error@+1:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sb a0, undefined, t0
# disabled-error@+1:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sh a0, undefined, t0
# disabled-error@+1:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension), 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sw a0, undefined, t0

# enabled-error@+2:22 {{too few operands for instruction}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sb a0, undefined

# enabled-error@+2:22 {{too few operands for instruction}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sh a0, undefined

# enabled-error@+2:22 {{too few operands for instruction}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sw a0, undefined

# enabled-error@+2:9 {{register must be a GPR excluding zero (x0)}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lb x0, undefined

# enabled-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lbu x0, undefined

# enabled-error@+2:9 {{register must be a GPR excluding zero (x0)}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lh x0, undefined

# enabled-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lhu x0, undefined

# enabled-error@+2:9 {{register must be a GPR excluding zero (x0)}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.lw x0, undefined

# enabled-error@+2:24 {{register must be a GPR excluding zero (x0)}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sb a0, undefined, x0

# enabled-error@+2:24 {{register must be a GPR excluding zero (x0)}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sh a0, undefined, x0

# enabled-error@+2:24 {{register must be a GPR excluding zero (x0)}}
# disabled-error@+1:1 {{instruction requires the following: 'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)}}
qc.e.sw a0, undefined, x0
