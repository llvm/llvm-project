// RUN: llvm-mc -triple=amdgpu12.50 -show-encoding %s | FileCheck --check-prefix=GFX1250 %s
// RUN: llvm-mc -triple=amdgpu12.50 -show-encoding %s | %extract-encodings | llvm-mc -triple=amdgpu12.50 -disassemble -show-encoding | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgpu12.00 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

s_load_b32 s4, s[2:3], 10 nv
// GFX1250: s_load_b32 s4, s[2:3], 0xa nv           ; encoding: [0x01,0x01,0x10,0xf4,0x0a,0x00,0x00,0xf8]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_load_b32 s4, s[2:3], 10 nv
// GFX12-ERR-NEXT:{{^}}                          ^

s_buffer_load_i8 s5, s[4:7], s0 nv
// GFX1250: s_buffer_load_i8 s5, s[4:7], s0 offset:0x0 nv ; encoding: [0x42,0x01,0x13,0xf4,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_buffer_load_i8 s5, s[4:7], s0 nv
// GFX12-ERR-NEXT:{{^}}                                ^

s_prefetch_inst s[12:13], 16, s4, 2 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX1250: s_prefetch_inst s[12:13], 0x10, s4, 2 th:TH_LOAD_HT scope:SCOPE_DEV nv ; encoding: [0x86,0x80,0x54,0xf5,0x10,0x00,0x00,0x08]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_prefetch_inst s[12:13], 16, s4, 2 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX12-ERR-NEXT:{{^}}                                                                  ^

s_prefetch_inst_pc_rel 100, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX1250: s_prefetch_inst_pc_rel 0x64, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv ; encoding: [0xc0,0xa1,0x54,0xf5,0x64,0x00,0x00,0x14]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_prefetch_inst_pc_rel 100, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX12-ERR-NEXT:{{^}}                                                                 ^

s_prefetch_data s[18:19], 100, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX1250: s_prefetch_data s[18:19], 0x64, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv ; encoding: [0xc9,0xc1,0x54,0xf5,0x64,0x00,0x00,0x14]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_prefetch_data s[18:19], 100, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX12-ERR-NEXT:{{^}}                                                                    ^

s_prefetch_data_pc_rel 100, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX1250: s_prefetch_data_pc_rel 0x64, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv ; encoding: [0xc0,0x01,0x55,0xf5,0x64,0x00,0x00,0x14]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_prefetch_data_pc_rel 100, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX12-ERR-NEXT:{{^}}                                                                 ^

s_buffer_prefetch_data s[8:11], 100, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX1250: s_buffer_prefetch_data s[8:11], 0x64, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv ; encoding: [0xc4,0xe1,0x54,0xf5,0x64,0x00,0x00,0x14]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_buffer_prefetch_data s[8:11], 100, s10, 7 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX12-ERR-NEXT:{{^}}                                                                          ^

s_atc_probe 7, s[4:5], s0 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX1250: s_atc_probe 7, s[4:5], s0 offset:0x0 th:TH_LOAD_HT scope:SCOPE_DEV nv ; encoding: [0xc2,0x41,0x54,0xf5,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_atc_probe 7, s[4:5], s0 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX12-ERR-NEXT:{{^}}                                                        ^

s_atc_probe_buffer 1, s[8:11], s0 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX1250: s_atc_probe_buffer 1, s[8:11], s0 offset:0x0 th:TH_LOAD_HT scope:SCOPE_DEV nv ; encoding: [0x44,0x60,0x54,0xf5,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_atc_probe_buffer 1, s[8:11], s0 th:TH_LOAD_HT scope:SCOPE_DEV nv
// GFX12-ERR-NEXT:{{^}}                                                                ^

s_load_b32 s4, s[2:3], 0xa scale_offset
// GFX1250: s_load_b32 s4, s[2:3], 0xa scale_offset ; encoding: [0x01,0x01,0x00,0xf4,0x0a,0x00,0x00,0xf9]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_load_b32 s4, s[2:3], 0xa scale_offset
// GFX12-ERR-NEXT:{{^}}                           ^

s_load_b32 s4, s[2:3], 0xa scale_offset nv
// GFX1250: s_load_b32 s4, s[2:3], 0xa scale_offset nv ; encoding: [0x01,0x01,0x10,0xf4,0x0a,0x00,0x00,0xf9]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_load_b32 s4, s[2:3], 0xa scale_offset nv
// GFX12-ERR-NEXT:{{^}}                           ^
// GFX12-ERR-NEXT: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_load_b32 s4, s[2:3], 0xa scale_offset nv
// GFX12-ERR-NEXT:{{^}}                                        ^

s_load_b32 s4, s[2:3], s5 offset:32 scale_offset
// GFX1250: s_load_b32 s4, s[2:3], s5 offset:0x20 scale_offset ; encoding: [0x01,0x01,0x00,0xf4,0x20,0x00,0x00,0x0b]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_load_b32 s4, s[2:3], s5 offset:32 scale_offset
// GFX12-ERR-NEXT:{{^}}                                    ^

s_load_b32 s4, s[2:3], m0 offset:32 scale_offset
// GFX1250: s_load_b32 s4, s[2:3], m0 offset:0x20 scale_offset ; encoding: [0x01,0x01,0x00,0xf4,0x20,0x00,0x00,0xfb]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}s_load_b32 s4, s[2:3], m0 offset:32 scale_offset
// GFX12-ERR-NEXT:{{^}}                                    ^
