// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck -check-prefix=GCN --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=GCN --implicit-check-not=error: %s

v_interp_p1_f32 v0, v1
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction

v_interp_p1_f32 v0, v1, attr64.w
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

v_interp_p1_f32 v0, v1, attr64.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

v_interp_p2_f32 v9, v1, attr64.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

v_interp_p2_f32 v0, v1, attr64.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

v_interp_p2_f32 v0, v1, attr0.q
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid or missing interpolation attribute channel

v_interp_p2_f32 v0, v1, attr0.
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid or missing interpolation attribute channel

v_interp_p2_f32 v0, v1, attr
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid or missing interpolation attribute channel

v_interp_p2_f32 v0, v1, att
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid interpolation attribute

v_interp_p2_f32 v0, v1, attrq
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid or missing interpolation attribute channel

v_interp_p2_f32 v7, v1, attr.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid or missing interpolation attribute number

v_interp_p2_f32 v7, v1, attr-1.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid or missing interpolation attribute channel

v_interp_p2_f32 v7, v1, attrA.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid or missing interpolation attribute number

v_interp_mov_f32 v11, invalid_param_3, attr0.y
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid interpolation slot

v_interp_mov_f32 v12, invalid_param_10, attr0.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid interpolation slot

v_interp_mov_f32 v3, invalid_param_3, attr0.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid interpolation slot

v_interp_mov_f32 v8, invalid_param_8, attr0.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid interpolation slot

v_interp_mov_f32 v8, foo, attr0.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid interpolation slot

v_interp_mov_f32 v8, 0, attr0.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_mov_f32 v8, -1, attr0.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_mov_f32 v8, p-1, attr0.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid interpolation slot

v_interp_mov_f32 v8, p1, attr0.x
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid interpolation slot
