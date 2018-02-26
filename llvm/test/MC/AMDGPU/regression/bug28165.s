// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SI --check-prefix=SICI
// RUN: llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SI --check-prefix=SICI
// RUN: llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI --check-prefix=CIVI
// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=CIVI --check-prefix=VI

// SICI: v_cmp_eq_f64_e32 vcc, 0.5, v[254:255] ; encoding: [0xf0,0xfc,0x45,0x7c]
// VI: v_cmp_eq_f64_e32 vcc, 0.5, v[254:255] ; encoding: [0xf0,0xfc,0xc5,0x7c]
v_cmp_eq_f64 vcc, 0.5, v[254:255]

// GCN: v_cvt_f32_f64_e32 v0, 0.5 ; encoding: [0xf0,0x1e,0x00,0x7e]
v_cvt_f32_f64 v0, 0.5
