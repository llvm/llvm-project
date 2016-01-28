// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s
// RUN: llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

s_nop 0       // CHECK: s_nop 0 ; encoding: [0x00,0x00,0x80,0xbf] 
s_nop 0xffff  // CHECK: s_nop 0xffff ; encoding: [0xff,0xff,0x80,0xbf]

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

  s_nop 1            // CHECK: s_nop 1 ; encoding: [0x01,0x00,0x80,0xbf]
  s_endpgm           // CHECK: s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]
  s_branch 2         // CHECK: s_branch 2 ; encoding: [0x02,0x00,0x82,0xbf]
  s_cbranch_scc0 3   // CHECK: s_cbranch_scc0 3 ; encoding: [0x03,0x00,0x84,0xbf]
  s_cbranch_scc1 4   // CHECK: s_cbranch_scc1 4 ; encoding: [0x04,0x00,0x85,0xbf]
  s_cbranch_vccz 5   // CHECK: s_cbranch_vccz 5 ; encoding: [0x05,0x00,0x86,0xbf]
  s_cbranch_vccnz 6  // CHECK: s_cbranch_vccnz 6 ; encoding: [0x06,0x00,0x87,0xbf]
  s_cbranch_execz 7  // CHECK: s_cbranch_execz 7 ; encoding: [0x07,0x00,0x88,0xbf]
  s_cbranch_execnz 8 // CHECK: s_cbranch_execnz 8 ; encoding: [0x08,0x00,0x89,0xbf]
  s_barrier          // CHECK: s_barrier ; encoding: [0x00,0x00,0x8a,0xbf]

//===----------------------------------------------------------------------===//
// s_waitcnt
//===----------------------------------------------------------------------===//

  s_waitcnt 0
  // CHECK: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

  s_waitcnt vmcnt(0) & expcnt(0) & lgkmcnt(0)
  // CHECK: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
  // CHECK: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

  s_waitcnt vmcnt(0), expcnt(0), lgkmcnt(0)
  // CHECK: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

  s_waitcnt vmcnt(1)
  // CHECK: s_waitcnt vmcnt(1) ; encoding: [0x71,0x0f,0x8c,0xbf]

  s_waitcnt vmcnt(9)
  // CHECK: s_waitcnt vmcnt(9) ; encoding: [0x79,0x0f,0x8c,0xbf]

  s_waitcnt expcnt(2)
  // CHECK: s_waitcnt expcnt(2) ; encoding: [0x2f,0x0f,0x8c,0xbf]

  s_waitcnt lgkmcnt(3)
  // CHECK: s_waitcnt lgkmcnt(3) ; encoding: [0x7f,0x03,0x8c,0xbf]

  s_waitcnt lgkmcnt(9)
  // CHECK: s_waitcnt lgkmcnt(9) ; encoding: [0x7f,0x09,0x8c,0xbf]

  s_waitcnt vmcnt(0), expcnt(0)
  // CHECK: s_waitcnt vmcnt(0) expcnt(0) ; encoding: [0x00,0x0f,0x8c,0xbf]


  s_sethalt 9        // CHECK: s_sethalt 9 ; encoding: [0x09,0x00,0x8d,0xbf]
  s_sleep 10         // CHECK: s_sleep 10 ; encoding: [0x0a,0x00,0x8e,0xbf]
  s_setprio 1        // CHECK: s_setprio 1 ; encoding: [0x01,0x00,0x8f,0xbf]
  s_sendmsg 2        // CHECK: s_sendmsg Gs(nop), [m0] ; encoding: [0x02,0x00,0x90,0xbf]
  s_sendmsghalt 3    // CHECK: s_sendmsghalt 3 ; encoding: [0x03,0x00,0x91,0xbf]
  s_trap 4           // CHECK: s_trap 4 ; encoding: [0x04,0x00,0x92,0xbf]
  s_icache_inv       // CHECK: s_icache_inv ; encoding: [0x00,0x00,0x93,0xbf]
  s_incperflevel 5   // CHECK: s_incperflevel 5 ; encoding: [0x05,0x00,0x94,0xbf]
  s_decperflevel 6   // CHECK: s_decperflevel 6 ; encoding: [0x06,0x00,0x95,0xbf]
  s_ttracedata       // CHECK: s_ttracedata ; encoding: [0x00,0x00,0x96,0xbf]
