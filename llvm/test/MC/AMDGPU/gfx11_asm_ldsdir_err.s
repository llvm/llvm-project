// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s 2>&1 | FileCheck -check-prefix=GFX11 %s

lds_param_load v17, attr33.x
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

lds_param_load v17, attr33.y
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

lds_param_load v17, attr33.z
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

lds_param_load v17, attr33.w
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

lds_param_load v12, attr33.z wait_va_vdst:4
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

lds_param_load v12, attr33.w wait_va_vdst:2 wait_vm_vsrc:1
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: out of bounds interpolation attribute number

