// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Vector Absolute Compare Mask Less Than Or Equal (Floating Point)
// FACLE is alias for FACGE with operands reversed
//----------------------------------------------------------------------
         facge v0.2s, v31.2s, v16.2s
         facge v4.4s, v7.4s, v15.4s
         facge v29.2d, v2.2d, v5.2d
         facle v0.2s, v16.2s, v31.2s
         facle v4.4s, v15.4s, v7.4s
         facle v29.2d, v5.2d, v2.2d

// CHECK: facge v0.2s, v31.2s, v16.2s // encoding: [0xe0,0xef,0x30,0x2e]
// CHECK: facge v4.4s, v7.4s, v15.4s  // encoding: [0xe4,0xec,0x2f,0x6e]
// CHECK: facge v29.2d, v2.2d, v5.2d  // encoding: [0x5d,0xec,0x65,0x6e]
// CHECK: facge v0.2s, v31.2s, v16.2s // encoding: [0xe0,0xef,0x30,0x2e]
// CHECK: facge v4.4s, v7.4s, v15.4s  // encoding: [0xe4,0xec,0x2f,0x6e]
// CHECK: facge v29.2d, v2.2d, v5.2d  // encoding: [0x5d,0xec,0x65,0x6e]

//----------------------------------------------------------------------
// Vector Absolute Compare Mask Less Than (Floating Point)
// FACLT is alias for FACGT with operands reversed
//----------------------------------------------------------------------
         facgt v31.4s, v29.4s, v28.4s
         facgt v3.2s, v8.2s, v12.2s
         facgt v17.2d, v15.2d, v13.2d
         faclt v31.4s, v28.4s, v29.4s
         faclt v3.2s,  v12.2s, v8.2s
         faclt v17.2d, v13.2d, v15.2d

// CHECK: facgt v31.4s, v29.4s, v28.4s  // encoding: [0xbf,0xef,0xbc,0x6e]
// CHECK: facgt v3.2s, v8.2s, v12.2s    // encoding: [0x03,0xed,0xac,0x2e]
// CHECK: facgt v17.2d, v15.2d, v13.2d  // encoding: [0xf1,0xed,0xed,0x6e]
// CHECK: facgt v31.4s, v29.4s, v28.4s  // encoding: [0xbf,0xef,0xbc,0x6e]
// CHECK: facgt v3.2s, v8.2s, v12.2s    // encoding: [0x03,0xed,0xac,0x2e]
// CHECK: facgt v17.2d, v15.2d, v13.2d  // encoding: [0xf1,0xed,0xed,0x6e]


