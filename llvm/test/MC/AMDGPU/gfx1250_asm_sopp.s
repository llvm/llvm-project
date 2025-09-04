// RUN: llvm-mc -triple=amdgcn -show-encoding -mcpu=gfx1250 %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: -strict-whitespace %s

s_wait_asynccnt 0x1234
// GFX1250: [0x34,0x12,0xca,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_asynccnt 0xc1d1
// GFX1250: [0xd1,0xc1,0xca,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_tensorcnt 0x0
// GFX1250: [0x00,0x00,0xcb,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_tensorcnt 0x1
// GFX1250: [0x01,0x00,0xcb,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_tensorcnt 0x3
// GFX1250: [0x03,0x00,0xcb,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_xcnt 0x0
// GFX1250: [0x00,0x00,0xc5,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_xcnt 0x7
// GFX1250: [0x07,0x00,0xc5,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_xcnt 0xf
// GFX1250: [0x0f,0x00,0xc5,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_setprio_inc_wg 100
// GFX1250: [0x64,0x00,0xbe,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_vgpr_msb 10
// GFX1250: [0x0a,0x00,0x86,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_set_vgpr_msb 255
// GFX1250: [0xff,0x00,0x86,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_monitor_sleep 1
// GFX1250: s_monitor_sleep 1                       ; encoding: [0x01,0x00,0x84,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_monitor_sleep 32768
// GFX1250: s_monitor_sleep 0x8000                  ; encoding: [0x00,0x80,0x84,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_monitor_sleep 0
// GFX1250: s_monitor_sleep 0                       ; encoding: [0x00,0x00,0x84,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_sendmsg sendmsg(MSG_SAVEWAVE_HAS_TDM)
// GFX1250: s_sendmsg sendmsg(MSG_SAVEWAVE_HAS_TDM)     ; encoding: [0x0a,0x00,0xb6,0xbf]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: specified message id is not supported on this GPU

s_barrier_wait -3
// GFX1250: s_barrier_wait -3                       ; encoding: [0xfd,0xff,0x94,0xbf]
