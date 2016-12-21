// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=SI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=VI %s

v_interp_p1_f32 v0, v1, attr64.w
// GCN: :25: error: out of bounds attr

v_interp_p1_f32 v0, v1, attr64.x
// GCN: :25: error: out of bounds attr

v_interp_p2_f32 v9, v1, attr64.x
// GCN: :25: error: out of bounds attr

v_interp_p2_f32 v0, v1, attr64.x
// GCN: :25: error: out of bounds attr

v_interp_p2_f32 v0, v1, attr0.q
// GCN: :25: error: failed parsing operand.

v_interp_p2_f32 v0, v1, attr0.
// GCN: :25: error: failed parsing operand.

v_interp_p2_f32 v0, v1, attr
// GCN: :25: error: failed parsing operand.

// XXX - Why does this one parse?
v_interp_p2_f32 v0, v1, att
// GCN: :25: error: invalid operand for instruction

v_interp_p2_f32 v0, v1, attrq
// GCN: :25: error: failed parsing operand.

v_interp_mov_f32 v11, invalid_param_3, attr0.y
// GCN: :23: error: failed parsing operand.

v_interp_mov_f32 v12, invalid_param_10, attr0.x
// GCN: :23: error: failed parsing operand.

v_interp_mov_f32 v3, invalid_param_3, attr0.x
// GCN: :22: error: failed parsing operand.

v_interp_mov_f32 v8, invalid_param_8, attr0.x
// GCN: :22: error: failed parsing operand.

v_interp_mov_f32 v8, foo, attr0.x
// GCN: :22: error: failed parsing operand.
