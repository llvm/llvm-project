// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize32 -show-encoding -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX13-W32-ERR --implicit-check-not=error: --strict-whitespace %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize64 -show-encoding -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX13-W64-ERR --strict-whitespace %s

v_send_vgpr_next_b32 s1, v2, v3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 s1, v2, v3
// GFX13-W32-ERR-NEXT:{{^}}                     ^
// GFX13-W64-ERR: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_send_vgpr_next_b32 v1, s2, v3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, s2, v3
// GFX13-W32-ERR-NEXT:{{^}}                         ^

v_send_vgpr_next_b32 v1, v2, s3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, s3
// GFX13-W32-ERR-NEXT:{{^}}                             ^

v_send_vgpr_next_b32 off, v2, v3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 off, v2, v3
// GFX13-W32-ERR-NEXT:{{^}}                          ^

v_send_vgpr_next_b32 v1, off, v3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, off, v3
// GFX13-W32-ERR-NEXT:{{^}}                         ^

v_send_vgpr_next_b32 v1, v2, off
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, off
// GFX13-W32-ERR-NEXT:{{^}}                             ^

v_send_vgpr_next_b32 v1, v2, v3 sema_id:8
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_id value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, v3 sema_id:8
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_next_b32 v1, v2, v3 sema_id:-5
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_id value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, v3 sema_id:-5
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_next_b32 v1, v2, v3 sema_wave_id:8
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_wave_id value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, v3 sema_wave_id:8
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_next_b32 v1, v2, v3 sema_wave_id:-5
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_wave_id value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, v3 sema_wave_id:-5
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_next_b32 v1, v2, v3 sema_id_refl:8
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_id_refl value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, v3 sema_id_refl:8
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_next_b32 v1, v2, v3 sema_id_refl:-5
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_id_refl value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, v3 sema_id_refl:-5
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_next_b32 v1, v2, v3 sema_wave_id_refl:8
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_wave_id_refl value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, v3 sema_wave_id_refl:8
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_next_b32 v1, v2, v3 sema_wave_id_refl:-5
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_wave_id_refl value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_next_b32 v1, v2, v3 sema_wave_id_refl:-5
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_prev_b32 s1, v2, v3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 s1, v2, v3
// GFX13-W32-ERR-NEXT:{{^}}                     ^
// GFX13-W64-ERR: :[[@LINE-4]]:{{[0-9]+}}: error: instruction requires wavesize=32

v_send_vgpr_prev_b32 v1, s2, v3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, s2, v3
// GFX13-W32-ERR-NEXT:{{^}}                         ^

v_send_vgpr_prev_b32 v1, v2, s3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, s3
// GFX13-W32-ERR-NEXT:{{^}}                             ^

v_send_vgpr_prev_b32 off, v2, v3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 off, v2, v3
// GFX13-W32-ERR-NEXT:{{^}}                          ^

v_send_vgpr_prev_b32 v1, off, v3
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, off, v3
// GFX13-W32-ERR-NEXT:{{^}}                         ^

v_send_vgpr_prev_b32 v1, v2, off
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, off
// GFX13-W32-ERR-NEXT:{{^}}                             ^

v_send_vgpr_prev_b32 v1, v2, v3 sema_id:8
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_id value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, v3 sema_id:8
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_prev_b32 v1, v2, v3 sema_id:-5
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_id value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, v3 sema_id:-5
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_prev_b32 v1, v2, v3 sema_wave_id:8
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_wave_id value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, v3 sema_wave_id:8
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_prev_b32 v1, v2, v3 sema_wave_id:-5
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_wave_id value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, v3 sema_wave_id:-5
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_prev_b32 v1, v2, v3 sema_id_refl:8
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_id_refl value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, v3 sema_id_refl:8
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_prev_b32 v1, v2, v3 sema_id_refl:-5
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_id_refl value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, v3 sema_id_refl:-5
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_prev_b32 v1, v2, v3 sema_wave_id_refl:8
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_wave_id_refl value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, v3 sema_wave_id_refl:8
// GFX13-W32-ERR-NEXT:{{^}}                                ^

v_send_vgpr_prev_b32 v1, v2, v3 sema_wave_id_refl:-5
// GFX13-W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid sema_wave_id_refl value.
// GFX13-W32-ERR-NEXT:{{^}}v_send_vgpr_prev_b32 v1, v2, v3 sema_wave_id_refl:-5
// GFX13-W32-ERR-NEXT:{{^}}                                ^
