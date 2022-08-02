; RUN: llc -march=amdgcn -mcpu=tahiti -enable-amdgpu-aa=0 -misched=gcn-iterative-minreg -verify-machineinstrs < %s | FileCheck --check-prefix=SI-MINREG %s
; RUN: llc -march=amdgcn -mcpu=tahiti -enable-amdgpu-aa=0 -misched=gcn-iterative-max-occupancy-experimental -verify-machineinstrs < %s | FileCheck --check-prefix=SI-MAXOCC %s
; RUN: llc -march=amdgcn -mcpu=fiji -enable-amdgpu-aa=0 -misched=gcn-iterative-minreg -verify-machineinstrs < %s | FileCheck --check-prefix=VI %s
; RUN: llc -march=amdgcn -mcpu=fiji -enable-amdgpu-aa=0 -misched=gcn-iterative-max-occupancy-experimental -verify-machineinstrs < %s | FileCheck --check-prefix=VI %s

; SI-MINREG: NumSgprs: {{[1-9]$}}
; SI-MINREG: NumVgprs: {{[1-9]$}}

; SI-MAXOCC: NumSgprs: {{[1-4]?[0-9]$}}
; SI-MAXOCC: NumVgprs: {{[1-4]?[0-9]$}}

; stores may alias loads
; VI: NumSgprs: {{[0-9]$}}
; VI: NumVgprs: {{[1-3][0-9]$}}

define amdgpu_kernel void @load_fma_store(float addrspace(3)* nocapture readonly %in_arg, float addrspace(1)* nocapture %out_arg) {
bb:
  %adr.a.0 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 20004
  %adr.b.0 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 20252
  %adr.c.0 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 20508
  %adr.a.1 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 20772
  %adr.b.1 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 21020
  %adr.c.1 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 21276
  %adr.a.2 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 21540
  %adr.b.2 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 21788
  %adr.c.2 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 22044
  %adr.a.3 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 22308
  %adr.b.3 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 22556
  %adr.c.3 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 22812
  %adr.a.4 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 23076
  %adr.b.4 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 23324
  %adr.c.4 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 23580
  %adr.a.5 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 23844
  %adr.b.5 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 24092
  %adr.c.5 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 24348
  %adr.a.6 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 24612
  %adr.b.6 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 24860
  %adr.c.6 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 25116
  %adr.a.7 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 25380
  %adr.b.7 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 25628
  %adr.c.7 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 25884
  %adr.a.8 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 26148
  %adr.b.8 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 26396
  %adr.c.8 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 26652
  %adr.a.9 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 26916
  %adr.b.9 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 27164
  %adr.c.9 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 27420
  %adr.a.10 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 27684
  %adr.b.10 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 27932
  %adr.c.10 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 28188
  %adr.a.11 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 28452
  %adr.b.11 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 28700
  %adr.c.11 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 28956
  %adr.a.12 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 29220
  %adr.b.12 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 29468
  %adr.c.12 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 29724
  %adr.a.13 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 29988
  %adr.b.13 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 30236
  %adr.c.13 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 30492
  %adr.a.14 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 30756
  %adr.b.14 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 31004
  %adr.c.14 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 31260
  %adr.a.15 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 31524
  %adr.b.15 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 31772
  %adr.c.15 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 32028
  %adr.a.16 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 32292
  %adr.b.16 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 32540
  %adr.c.16 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 32796
  %adr.a.17 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 33060
  %adr.b.17 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 33308
  %adr.c.17 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 33564
  %adr.a.18 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 33828
  %adr.b.18 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 34076
  %adr.c.18 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 34332
  %adr.a.19 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 34596
  %adr.b.19 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 34844
  %adr.c.19 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 35100
  %adr.a.20 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 35364
  %adr.b.20 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 35612
  %adr.c.20 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 35868
  %adr.a.21 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 36132
  %adr.b.21 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 36380
  %adr.c.21 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 36636
  %adr.a.22 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 36900
  %adr.b.22 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 37148
  %adr.c.22 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 37404
  %adr.a.23 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 37668
  %adr.b.23 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 37916
  %adr.c.23 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 38172
  %adr.a.24 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 38436
  %adr.b.24 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 38684
  %adr.c.24 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 38940
  %adr.a.25 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 39204
  %adr.b.25 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 39452
  %adr.c.25 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 39708
  %adr.a.26 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 39972
  %adr.b.26 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 40220
  %adr.c.26 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 40476
  %adr.a.27 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 40740
  %adr.b.27 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 40988
  %adr.c.27 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 41244
  %adr.a.28 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 41508
  %adr.b.28 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 41756
  %adr.c.28 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 42012
  %adr.a.29 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 42276
  %adr.b.29 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 42524
  %adr.c.29 = getelementptr inbounds float, float addrspace(3)* %in_arg, i32 42780
  %a.0 = load float, float addrspace(3)* %adr.a.0, align 4
  %b.0 = load float, float addrspace(3)* %adr.b.0, align 4
  %c.0 = load float, float addrspace(3)* %adr.c.0, align 4
  %a.1 = load float, float addrspace(3)* %adr.a.1, align 4
  %b.1 = load float, float addrspace(3)* %adr.b.1, align 4
  %c.1 = load float, float addrspace(3)* %adr.c.1, align 4
  %a.2 = load float, float addrspace(3)* %adr.a.2, align 4
  %b.2 = load float, float addrspace(3)* %adr.b.2, align 4
  %c.2 = load float, float addrspace(3)* %adr.c.2, align 4
  %a.3 = load float, float addrspace(3)* %adr.a.3, align 4
  %b.3 = load float, float addrspace(3)* %adr.b.3, align 4
  %c.3 = load float, float addrspace(3)* %adr.c.3, align 4
  %a.4 = load float, float addrspace(3)* %adr.a.4, align 4
  %b.4 = load float, float addrspace(3)* %adr.b.4, align 4
  %c.4 = load float, float addrspace(3)* %adr.c.4, align 4
  %a.5 = load float, float addrspace(3)* %adr.a.5, align 4
  %b.5 = load float, float addrspace(3)* %adr.b.5, align 4
  %c.5 = load float, float addrspace(3)* %adr.c.5, align 4
  %a.6 = load float, float addrspace(3)* %adr.a.6, align 4
  %b.6 = load float, float addrspace(3)* %adr.b.6, align 4
  %c.6 = load float, float addrspace(3)* %adr.c.6, align 4
  %a.7 = load float, float addrspace(3)* %adr.a.7, align 4
  %b.7 = load float, float addrspace(3)* %adr.b.7, align 4
  %c.7 = load float, float addrspace(3)* %adr.c.7, align 4
  %a.8 = load float, float addrspace(3)* %adr.a.8, align 4
  %b.8 = load float, float addrspace(3)* %adr.b.8, align 4
  %c.8 = load float, float addrspace(3)* %adr.c.8, align 4
  %a.9 = load float, float addrspace(3)* %adr.a.9, align 4
  %b.9 = load float, float addrspace(3)* %adr.b.9, align 4
  %c.9 = load float, float addrspace(3)* %adr.c.9, align 4
  %a.10 = load float, float addrspace(3)* %adr.a.10, align 4
  %b.10 = load float, float addrspace(3)* %adr.b.10, align 4
  %c.10 = load float, float addrspace(3)* %adr.c.10, align 4
  %a.11 = load float, float addrspace(3)* %adr.a.11, align 4
  %b.11 = load float, float addrspace(3)* %adr.b.11, align 4
  %c.11 = load float, float addrspace(3)* %adr.c.11, align 4
  %a.12 = load float, float addrspace(3)* %adr.a.12, align 4
  %b.12 = load float, float addrspace(3)* %adr.b.12, align 4
  %c.12 = load float, float addrspace(3)* %adr.c.12, align 4
  %a.13 = load float, float addrspace(3)* %adr.a.13, align 4
  %b.13 = load float, float addrspace(3)* %adr.b.13, align 4
  %c.13 = load float, float addrspace(3)* %adr.c.13, align 4
  %a.14 = load float, float addrspace(3)* %adr.a.14, align 4
  %b.14 = load float, float addrspace(3)* %adr.b.14, align 4
  %c.14 = load float, float addrspace(3)* %adr.c.14, align 4
  %a.15 = load float, float addrspace(3)* %adr.a.15, align 4
  %b.15 = load float, float addrspace(3)* %adr.b.15, align 4
  %c.15 = load float, float addrspace(3)* %adr.c.15, align 4
  %a.16 = load float, float addrspace(3)* %adr.a.16, align 4
  %b.16 = load float, float addrspace(3)* %adr.b.16, align 4
  %c.16 = load float, float addrspace(3)* %adr.c.16, align 4
  %a.17 = load float, float addrspace(3)* %adr.a.17, align 4
  %b.17 = load float, float addrspace(3)* %adr.b.17, align 4
  %c.17 = load float, float addrspace(3)* %adr.c.17, align 4
  %a.18 = load float, float addrspace(3)* %adr.a.18, align 4
  %b.18 = load float, float addrspace(3)* %adr.b.18, align 4
  %c.18 = load float, float addrspace(3)* %adr.c.18, align 4
  %a.19 = load float, float addrspace(3)* %adr.a.19, align 4
  %b.19 = load float, float addrspace(3)* %adr.b.19, align 4
  %c.19 = load float, float addrspace(3)* %adr.c.19, align 4
  %a.20 = load float, float addrspace(3)* %adr.a.20, align 4
  %b.20 = load float, float addrspace(3)* %adr.b.20, align 4
  %c.20 = load float, float addrspace(3)* %adr.c.20, align 4
  %a.21 = load float, float addrspace(3)* %adr.a.21, align 4
  %b.21 = load float, float addrspace(3)* %adr.b.21, align 4
  %c.21 = load float, float addrspace(3)* %adr.c.21, align 4
  %a.22 = load float, float addrspace(3)* %adr.a.22, align 4
  %b.22 = load float, float addrspace(3)* %adr.b.22, align 4
  %c.22 = load float, float addrspace(3)* %adr.c.22, align 4
  %a.23 = load float, float addrspace(3)* %adr.a.23, align 4
  %b.23 = load float, float addrspace(3)* %adr.b.23, align 4
  %c.23 = load float, float addrspace(3)* %adr.c.23, align 4
  %a.24 = load float, float addrspace(3)* %adr.a.24, align 4
  %b.24 = load float, float addrspace(3)* %adr.b.24, align 4
  %c.24 = load float, float addrspace(3)* %adr.c.24, align 4
  %a.25 = load float, float addrspace(3)* %adr.a.25, align 4
  %b.25 = load float, float addrspace(3)* %adr.b.25, align 4
  %c.25 = load float, float addrspace(3)* %adr.c.25, align 4
  %a.26 = load float, float addrspace(3)* %adr.a.26, align 4
  %b.26 = load float, float addrspace(3)* %adr.b.26, align 4
  %c.26 = load float, float addrspace(3)* %adr.c.26, align 4
  %a.27 = load float, float addrspace(3)* %adr.a.27, align 4
  %b.27 = load float, float addrspace(3)* %adr.b.27, align 4
  %c.27 = load float, float addrspace(3)* %adr.c.27, align 4
  %a.28 = load float, float addrspace(3)* %adr.a.28, align 4
  %b.28 = load float, float addrspace(3)* %adr.b.28, align 4
  %c.28 = load float, float addrspace(3)* %adr.c.28, align 4
  %a.29 = load float, float addrspace(3)* %adr.a.29, align 4
  %b.29 = load float, float addrspace(3)* %adr.b.29, align 4
  %c.29 = load float, float addrspace(3)* %adr.c.29, align 4
  %res.0 = tail call float @llvm.fmuladd.f32(float %a.0, float %b.0, float %c.0)
  %res.1 = tail call float @llvm.fmuladd.f32(float %a.1, float %b.1, float %c.1)
  %res.2 = tail call float @llvm.fmuladd.f32(float %a.2, float %b.2, float %c.2)
  %res.3 = tail call float @llvm.fmuladd.f32(float %a.3, float %b.3, float %c.3)
  %res.4 = tail call float @llvm.fmuladd.f32(float %a.4, float %b.4, float %c.4)
  %res.5 = tail call float @llvm.fmuladd.f32(float %a.5, float %b.5, float %c.5)
  %res.6 = tail call float @llvm.fmuladd.f32(float %a.6, float %b.6, float %c.6)
  %res.7 = tail call float @llvm.fmuladd.f32(float %a.7, float %b.7, float %c.7)
  %res.8 = tail call float @llvm.fmuladd.f32(float %a.8, float %b.8, float %c.8)
  %res.9 = tail call float @llvm.fmuladd.f32(float %a.9, float %b.9, float %c.9)
  %res.10 = tail call float @llvm.fmuladd.f32(float %a.10, float %b.10, float %c.10)
  %res.11 = tail call float @llvm.fmuladd.f32(float %a.11, float %b.11, float %c.11)
  %res.12 = tail call float @llvm.fmuladd.f32(float %a.12, float %b.12, float %c.12)
  %res.13 = tail call float @llvm.fmuladd.f32(float %a.13, float %b.13, float %c.13)
  %res.14 = tail call float @llvm.fmuladd.f32(float %a.14, float %b.14, float %c.14)
  %res.15 = tail call float @llvm.fmuladd.f32(float %a.15, float %b.15, float %c.15)
  %res.16 = tail call float @llvm.fmuladd.f32(float %a.16, float %b.16, float %c.16)
  %res.17 = tail call float @llvm.fmuladd.f32(float %a.17, float %b.17, float %c.17)
  %res.18 = tail call float @llvm.fmuladd.f32(float %a.18, float %b.18, float %c.18)
  %res.19 = tail call float @llvm.fmuladd.f32(float %a.19, float %b.19, float %c.19)
  %res.20 = tail call float @llvm.fmuladd.f32(float %a.20, float %b.20, float %c.20)
  %res.21 = tail call float @llvm.fmuladd.f32(float %a.21, float %b.21, float %c.21)
  %res.22 = tail call float @llvm.fmuladd.f32(float %a.22, float %b.22, float %c.22)
  %res.23 = tail call float @llvm.fmuladd.f32(float %a.23, float %b.23, float %c.23)
  %res.24 = tail call float @llvm.fmuladd.f32(float %a.24, float %b.24, float %c.24)
  %res.25 = tail call float @llvm.fmuladd.f32(float %a.25, float %b.25, float %c.25)
  %res.26 = tail call float @llvm.fmuladd.f32(float %a.26, float %b.26, float %c.26)
  %res.27 = tail call float @llvm.fmuladd.f32(float %a.27, float %b.27, float %c.27)
  %res.28 = tail call float @llvm.fmuladd.f32(float %a.28, float %b.28, float %c.28)
  %res.29 = tail call float @llvm.fmuladd.f32(float %a.29, float %b.29, float %c.29)
  %adr.res.0 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 0
  %adr.res.1 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 2
  %adr.res.2 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 4
  %adr.res.3 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 6
  %adr.res.4 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 8
  %adr.res.5 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 10
  %adr.res.6 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 12
  %adr.res.7 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 14
  %adr.res.8 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 16
  %adr.res.9 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 18
  %adr.res.10 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 20
  %adr.res.11 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 22
  %adr.res.12 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 24
  %adr.res.13 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 26
  %adr.res.14 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 28
  %adr.res.15 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 30
  %adr.res.16 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 32
  %adr.res.17 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 34
  %adr.res.18 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 36
  %adr.res.19 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 38
  %adr.res.20 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 40
  %adr.res.21 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 42
  %adr.res.22 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 44
  %adr.res.23 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 46
  %adr.res.24 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 48
  %adr.res.25 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 50
  %adr.res.26 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 52
  %adr.res.27 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 54
  %adr.res.28 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 56
  %adr.res.29 = getelementptr inbounds float, float addrspace(1)* %out_arg, i64 58
  store float %res.0, float addrspace(1)* %adr.res.0, align 4
  store float %res.1, float addrspace(1)* %adr.res.1, align 4
  store float %res.2, float addrspace(1)* %adr.res.2, align 4
  store float %res.3, float addrspace(1)* %adr.res.3, align 4
  store float %res.4, float addrspace(1)* %adr.res.4, align 4
  store float %res.5, float addrspace(1)* %adr.res.5, align 4
  store float %res.6, float addrspace(1)* %adr.res.6, align 4
  store float %res.7, float addrspace(1)* %adr.res.7, align 4
  store float %res.8, float addrspace(1)* %adr.res.8, align 4
  store float %res.9, float addrspace(1)* %adr.res.9, align 4
  store float %res.10, float addrspace(1)* %adr.res.10, align 4
  store float %res.11, float addrspace(1)* %adr.res.11, align 4
  store float %res.12, float addrspace(1)* %adr.res.12, align 4
  store float %res.13, float addrspace(1)* %adr.res.13, align 4
  store float %res.14, float addrspace(1)* %adr.res.14, align 4
  store float %res.15, float addrspace(1)* %adr.res.15, align 4
  store float %res.16, float addrspace(1)* %adr.res.16, align 4
  store float %res.17, float addrspace(1)* %adr.res.17, align 4
  store float %res.18, float addrspace(1)* %adr.res.18, align 4
  store float %res.19, float addrspace(1)* %adr.res.19, align 4
  store float %res.20, float addrspace(1)* %adr.res.20, align 4
  store float %res.21, float addrspace(1)* %adr.res.21, align 4
  store float %res.22, float addrspace(1)* %adr.res.22, align 4
  store float %res.23, float addrspace(1)* %adr.res.23, align 4
  store float %res.24, float addrspace(1)* %adr.res.24, align 4
  store float %res.25, float addrspace(1)* %adr.res.25, align 4
  store float %res.26, float addrspace(1)* %adr.res.26, align 4
  store float %res.27, float addrspace(1)* %adr.res.27, align 4
  store float %res.28, float addrspace(1)* %adr.res.28, align 4
  store float %res.29, float addrspace(1)* %adr.res.29, align 4
  ret void
}
declare float @llvm.fmuladd.f32(float, float, float) #0
attributes #0 = { nounwind readnone }
