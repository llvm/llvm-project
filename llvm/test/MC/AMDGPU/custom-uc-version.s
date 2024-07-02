// RUN: llvm-mc -triple=amdgcn -show-encoding -mcpu=gfx1010 %s | FileCheck --check-prefix=GFX10 %s

// Test that UC_VERSION* symbols can be redefined.

.set UC_VERSION_GFX10, 99

s_version UC_VERSION_GFX10
// GFX10: encoding: [0x63,0x00,0x80,0xb0]

s_version UC_VERSION_GFX10 | UC_VERSION_W32_BIT
// GFX10: encoding: [0x63,0x40,0x80,0xb0]

s_version UC_VERSION_GFX10 | UC_VERSION_W64_BIT
// GFX10: encoding: [0x63,0x20,0x80,0xb0]

s_version UC_VERSION_GFX10 | UC_VERSION_MDP_BIT
// GFX10: encoding: [0x63,0x80,0x80,0xb0]

s_version UC_VERSION_GFX10 | UC_VERSION_W64_BIT | UC_VERSION_MDP_BIT
// GFX10: encoding: [0x63,0xa0,0x80,0xb0]

.set UC_VERSION_GFX10, 100
.set UC_VERSION_W32_BIT, 0
.set UC_VERSION_W64_BIT, 0
.set UC_VERSION_MDP_BIT, 0

s_version UC_VERSION_GFX10
// GFX10: encoding: [0x64,0x00,0x80,0xb0]

s_version UC_VERSION_GFX10 | UC_VERSION_W32_BIT
// GFX10: encoding: [0x64,0x00,0x80,0xb0]

s_version UC_VERSION_GFX10 | UC_VERSION_W64_BIT
// GFX10: encoding: [0x64,0x00,0x80,0xb0]

s_version UC_VERSION_GFX10 | UC_VERSION_MDP_BIT
// GFX10: encoding: [0x64,0x00,0x80,0xb0]

s_version UC_VERSION_GFX10 | UC_VERSION_W64_BIT | UC_VERSION_MDP_BIT
// GFX10: encoding: [0x64,0x00,0x80,0xb0]

