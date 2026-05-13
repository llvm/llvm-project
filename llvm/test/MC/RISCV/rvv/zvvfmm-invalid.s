# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvvfmm %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vfmmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: vfmmacc.vv v8, v4, v20, v0.t

# vm=0 is reserved for non-widening vfmmacc.vv.
vfmmacc.vv v8, v4, v20, v0.scale
# CHECK-ERROR: invalid operand for instruction
# CHECK-ERROR-LABEL: vfmmacc.vv v8, v4, v20, v0.scale

vfwmmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: expected '.scale' suffix
# CHECK-ERROR-LABEL: vfwmmacc.vv v8, v4, v20, v0.t

vfwmmacc.vv v8, v4, v20, v1.scale
# CHECK-ERROR: operand must be v0.scale
# CHECK-ERROR-LABEL: vfwmmacc.vv v8, v4, v20, v1.scale

vfqmmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: expected '.scale' suffix
# CHECK-ERROR-LABEL: vfqmmacc.vv v8, v4, v20, v0.t

vfqmmacc.vv v8, v4, v20, v1.scale
# CHECK-ERROR: operand must be v0.scale
# CHECK-ERROR-LABEL: vfqmmacc.vv v8, v4, v20, v1.scale

vf8wmmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: expected '.scale' suffix
# CHECK-ERROR-LABEL: vf8wmmacc.vv v8, v4, v20, v0.t

vf8wmmacc.vv v8, v4, v20, v1.scale
# CHECK-ERROR: operand must be v0.scale
# CHECK-ERROR-LABEL: vf8wmmacc.vv v8, v4, v20, v1.scale

vfwmmacc.vv v0, v4, v20, v0.scale
# CHECK-ERROR: vd, vs1, and vs2 cannot overlap v0.scale
# CHECK-ERROR-LABEL: vfwmmacc.vv v0, v4, v20, v0.scale

vfqmmacc.vv v8, v0, v20, v0.scale
# CHECK-ERROR: vd, vs1, and vs2 cannot overlap v0.scale
# CHECK-ERROR-LABEL: vfqmmacc.vv v8, v0, v20, v0.scale

vf8wmmacc.vv v8, v4, v0, v0.scale
# CHECK-ERROR: vd, vs1, and vs2 cannot overlap v0.scale
# CHECK-ERROR-LABEL: vf8wmmacc.vv v8, v4, v0, v0.scale

vfwimmacc.vv v8, v4, v20
# CHECK-ERROR: too few operands for instruction
# CHECK-ERROR-LABEL: vfwimmacc.vv v8, v4, v20

vfwimmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: expected '.scale' suffix
# CHECK-ERROR-LABEL: vfwimmacc.vv v8, v4, v20, v0.t

vfwimmacc.vv v8, v4, v20, v1.scale
# CHECK-ERROR: operand must be v0.scale
# CHECK-ERROR-LABEL: vfwimmacc.vv v8, v4, v20, v1.scale

vfwimmacc.vv v0, v4, v20, v0.scale
# CHECK-ERROR: vd, vs1, and vs2 cannot overlap v0.scale
# CHECK-ERROR-LABEL: vfwimmacc.vv v0, v4, v20, v0.scale

vfqimmacc.vv v8, v4, v20
# CHECK-ERROR: too few operands for instruction
# CHECK-ERROR-LABEL: vfqimmacc.vv v8, v4, v20

vfqimmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: expected '.scale' suffix
# CHECK-ERROR-LABEL: vfqimmacc.vv v8, v4, v20, v0.t

vfqimmacc.vv v8, v4, v20, v1.scale
# CHECK-ERROR: operand must be v0.scale
# CHECK-ERROR-LABEL: vfqimmacc.vv v8, v4, v20, v1.scale

vfqimmacc.vv v8, v0, v20, v0.scale
# CHECK-ERROR: vd, vs1, and vs2 cannot overlap v0.scale
# CHECK-ERROR-LABEL: vfqimmacc.vv v8, v0, v20, v0.scale

vf8wimmacc.vv v8, v4, v20
# CHECK-ERROR: too few operands for instruction
# CHECK-ERROR-LABEL: vf8wimmacc.vv v8, v4, v20

vf8wimmacc.vv v8, v4, v20, v0.t
# CHECK-ERROR: expected '.scale' suffix
# CHECK-ERROR-LABEL: vf8wimmacc.vv v8, v4, v20, v0.t

vf8wimmacc.vv v8, v4, v20, v1.scale
# CHECK-ERROR: operand must be v0.scale
# CHECK-ERROR-LABEL: vf8wimmacc.vv v8, v4, v20, v1.scale

vf8wimmacc.vv v8, v4, v0, v0.scale
# CHECK-ERROR: vd, vs1, and vs2 cannot overlap v0.scale
# CHECK-ERROR-LABEL: vf8wimmacc.vv v8, v4, v0, v0.scale
