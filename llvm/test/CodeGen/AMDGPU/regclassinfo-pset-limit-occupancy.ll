; REQUIRES: asserts
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -enable-new-pm=0 \
; RUN:   -debug-only=machine-scheduler -filetype=null %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -enable-new-pm=1 \
; RUN:   -debug-only=machine-scheduler -filetype=null %s 2>&1 | FileCheck %s
;
; RegisterClassInfo caches target register pressure limits across functions.
; AMDGPU derives these limits from LDS-dependent occupancy, which can change
; without changing the reserved or callee-saved registers used to invalidate
; the cache. Check that the second kernel gets its own pressure limit.
;
; CHECK: # Machine code for function kernel_small:
; CHECK: Starting occupancy is 8.
; CHECK: VGPR_32 Limit 64 Actual {{[0-9]+}}
; CHECK: VGPRCriticalLimit = 61,
; CHECK: # Machine code for function kernel_large:
; CHECK: Starting occupancy is 1.
; CHECK-NOT: VGPR_32 Limit
; CHECK: VGPRCriticalLimit = 253,

@lds.small = internal addrspace(3) global [16 x i32] poison, align 16
@lds.large = internal addrspace(3) global [16384 x i32] poison, align 16

declare i32 @llvm.amdgcn.workitem.id.x()

define internal void @create_pressure(ptr addrspace(1) %in,
                                      ptr addrspace(1) %out) alwaysinline {
entry:
  %p0 = getelementptr inbounds float, ptr addrspace(1) %in, i64 0
  %v0 = load volatile float, ptr addrspace(1) %p0, align 4
  %p1 = getelementptr inbounds float, ptr addrspace(1) %in, i64 1
  %v1 = load volatile float, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr inbounds float, ptr addrspace(1) %in, i64 2
  %v2 = load volatile float, ptr addrspace(1) %p2, align 4
  %p3 = getelementptr inbounds float, ptr addrspace(1) %in, i64 3
  %v3 = load volatile float, ptr addrspace(1) %p3, align 4
  %p4 = getelementptr inbounds float, ptr addrspace(1) %in, i64 4
  %v4 = load volatile float, ptr addrspace(1) %p4, align 4
  %p5 = getelementptr inbounds float, ptr addrspace(1) %in, i64 5
  %v5 = load volatile float, ptr addrspace(1) %p5, align 4
  %p6 = getelementptr inbounds float, ptr addrspace(1) %in, i64 6
  %v6 = load volatile float, ptr addrspace(1) %p6, align 4
  %p7 = getelementptr inbounds float, ptr addrspace(1) %in, i64 7
  %v7 = load volatile float, ptr addrspace(1) %p7, align 4
  %p8 = getelementptr inbounds float, ptr addrspace(1) %in, i64 8
  %v8 = load volatile float, ptr addrspace(1) %p8, align 4
  %p9 = getelementptr inbounds float, ptr addrspace(1) %in, i64 9
  %v9 = load volatile float, ptr addrspace(1) %p9, align 4
  %p10 = getelementptr inbounds float, ptr addrspace(1) %in, i64 10
  %v10 = load volatile float, ptr addrspace(1) %p10, align 4
  %p11 = getelementptr inbounds float, ptr addrspace(1) %in, i64 11
  %v11 = load volatile float, ptr addrspace(1) %p11, align 4
  %p12 = getelementptr inbounds float, ptr addrspace(1) %in, i64 12
  %v12 = load volatile float, ptr addrspace(1) %p12, align 4
  %p13 = getelementptr inbounds float, ptr addrspace(1) %in, i64 13
  %v13 = load volatile float, ptr addrspace(1) %p13, align 4
  %p14 = getelementptr inbounds float, ptr addrspace(1) %in, i64 14
  %v14 = load volatile float, ptr addrspace(1) %p14, align 4
  %p15 = getelementptr inbounds float, ptr addrspace(1) %in, i64 15
  %v15 = load volatile float, ptr addrspace(1) %p15, align 4
  %p16 = getelementptr inbounds float, ptr addrspace(1) %in, i64 16
  %v16 = load volatile float, ptr addrspace(1) %p16, align 4
  %p17 = getelementptr inbounds float, ptr addrspace(1) %in, i64 17
  %v17 = load volatile float, ptr addrspace(1) %p17, align 4
  %p18 = getelementptr inbounds float, ptr addrspace(1) %in, i64 18
  %v18 = load volatile float, ptr addrspace(1) %p18, align 4
  %p19 = getelementptr inbounds float, ptr addrspace(1) %in, i64 19
  %v19 = load volatile float, ptr addrspace(1) %p19, align 4
  %p20 = getelementptr inbounds float, ptr addrspace(1) %in, i64 20
  %v20 = load volatile float, ptr addrspace(1) %p20, align 4
  %p21 = getelementptr inbounds float, ptr addrspace(1) %in, i64 21
  %v21 = load volatile float, ptr addrspace(1) %p21, align 4
  %p22 = getelementptr inbounds float, ptr addrspace(1) %in, i64 22
  %v22 = load volatile float, ptr addrspace(1) %p22, align 4
  %p23 = getelementptr inbounds float, ptr addrspace(1) %in, i64 23
  %v23 = load volatile float, ptr addrspace(1) %p23, align 4
  %p24 = getelementptr inbounds float, ptr addrspace(1) %in, i64 24
  %v24 = load volatile float, ptr addrspace(1) %p24, align 4
  %p25 = getelementptr inbounds float, ptr addrspace(1) %in, i64 25
  %v25 = load volatile float, ptr addrspace(1) %p25, align 4
  %p26 = getelementptr inbounds float, ptr addrspace(1) %in, i64 26
  %v26 = load volatile float, ptr addrspace(1) %p26, align 4
  %p27 = getelementptr inbounds float, ptr addrspace(1) %in, i64 27
  %v27 = load volatile float, ptr addrspace(1) %p27, align 4
  %p28 = getelementptr inbounds float, ptr addrspace(1) %in, i64 28
  %v28 = load volatile float, ptr addrspace(1) %p28, align 4
  %p29 = getelementptr inbounds float, ptr addrspace(1) %in, i64 29
  %v29 = load volatile float, ptr addrspace(1) %p29, align 4
  %p30 = getelementptr inbounds float, ptr addrspace(1) %in, i64 30
  %v30 = load volatile float, ptr addrspace(1) %p30, align 4
  %p31 = getelementptr inbounds float, ptr addrspace(1) %in, i64 31
  %v31 = load volatile float, ptr addrspace(1) %p31, align 4
  %p32 = getelementptr inbounds float, ptr addrspace(1) %in, i64 32
  %v32 = load volatile float, ptr addrspace(1) %p32, align 4
  %p33 = getelementptr inbounds float, ptr addrspace(1) %in, i64 33
  %v33 = load volatile float, ptr addrspace(1) %p33, align 4
  %p34 = getelementptr inbounds float, ptr addrspace(1) %in, i64 34
  %v34 = load volatile float, ptr addrspace(1) %p34, align 4
  %p35 = getelementptr inbounds float, ptr addrspace(1) %in, i64 35
  %v35 = load volatile float, ptr addrspace(1) %p35, align 4
  %p36 = getelementptr inbounds float, ptr addrspace(1) %in, i64 36
  %v36 = load volatile float, ptr addrspace(1) %p36, align 4
  %p37 = getelementptr inbounds float, ptr addrspace(1) %in, i64 37
  %v37 = load volatile float, ptr addrspace(1) %p37, align 4
  %p38 = getelementptr inbounds float, ptr addrspace(1) %in, i64 38
  %v38 = load volatile float, ptr addrspace(1) %p38, align 4
  %p39 = getelementptr inbounds float, ptr addrspace(1) %in, i64 39
  %v39 = load volatile float, ptr addrspace(1) %p39, align 4
  %p40 = getelementptr inbounds float, ptr addrspace(1) %in, i64 40
  %v40 = load volatile float, ptr addrspace(1) %p40, align 4
  %p41 = getelementptr inbounds float, ptr addrspace(1) %in, i64 41
  %v41 = load volatile float, ptr addrspace(1) %p41, align 4
  %p42 = getelementptr inbounds float, ptr addrspace(1) %in, i64 42
  %v42 = load volatile float, ptr addrspace(1) %p42, align 4
  %p43 = getelementptr inbounds float, ptr addrspace(1) %in, i64 43
  %v43 = load volatile float, ptr addrspace(1) %p43, align 4
  %p44 = getelementptr inbounds float, ptr addrspace(1) %in, i64 44
  %v44 = load volatile float, ptr addrspace(1) %p44, align 4
  %p45 = getelementptr inbounds float, ptr addrspace(1) %in, i64 45
  %v45 = load volatile float, ptr addrspace(1) %p45, align 4
  %p46 = getelementptr inbounds float, ptr addrspace(1) %in, i64 46
  %v46 = load volatile float, ptr addrspace(1) %p46, align 4
  %p47 = getelementptr inbounds float, ptr addrspace(1) %in, i64 47
  %v47 = load volatile float, ptr addrspace(1) %p47, align 4
  %p48 = getelementptr inbounds float, ptr addrspace(1) %in, i64 48
  %v48 = load volatile float, ptr addrspace(1) %p48, align 4
  %p49 = getelementptr inbounds float, ptr addrspace(1) %in, i64 49
  %v49 = load volatile float, ptr addrspace(1) %p49, align 4
  %p50 = getelementptr inbounds float, ptr addrspace(1) %in, i64 50
  %v50 = load volatile float, ptr addrspace(1) %p50, align 4
  %p51 = getelementptr inbounds float, ptr addrspace(1) %in, i64 51
  %v51 = load volatile float, ptr addrspace(1) %p51, align 4
  %p52 = getelementptr inbounds float, ptr addrspace(1) %in, i64 52
  %v52 = load volatile float, ptr addrspace(1) %p52, align 4
  %p53 = getelementptr inbounds float, ptr addrspace(1) %in, i64 53
  %v53 = load volatile float, ptr addrspace(1) %p53, align 4
  %p54 = getelementptr inbounds float, ptr addrspace(1) %in, i64 54
  %v54 = load volatile float, ptr addrspace(1) %p54, align 4
  %p55 = getelementptr inbounds float, ptr addrspace(1) %in, i64 55
  %v55 = load volatile float, ptr addrspace(1) %p55, align 4
  %p56 = getelementptr inbounds float, ptr addrspace(1) %in, i64 56
  %v56 = load volatile float, ptr addrspace(1) %p56, align 4
  %p57 = getelementptr inbounds float, ptr addrspace(1) %in, i64 57
  %v57 = load volatile float, ptr addrspace(1) %p57, align 4
  %p58 = getelementptr inbounds float, ptr addrspace(1) %in, i64 58
  %v58 = load volatile float, ptr addrspace(1) %p58, align 4
  %p59 = getelementptr inbounds float, ptr addrspace(1) %in, i64 59
  %v59 = load volatile float, ptr addrspace(1) %p59, align 4
  %p60 = getelementptr inbounds float, ptr addrspace(1) %in, i64 60
  %v60 = load volatile float, ptr addrspace(1) %p60, align 4
  %p61 = getelementptr inbounds float, ptr addrspace(1) %in, i64 61
  %v61 = load volatile float, ptr addrspace(1) %p61, align 4
  %p62 = getelementptr inbounds float, ptr addrspace(1) %in, i64 62
  %v62 = load volatile float, ptr addrspace(1) %p62, align 4
  %p63 = getelementptr inbounds float, ptr addrspace(1) %in, i64 63
  %v63 = load volatile float, ptr addrspace(1) %p63, align 4
  %p64 = getelementptr inbounds float, ptr addrspace(1) %in, i64 64
  %v64 = load volatile float, ptr addrspace(1) %p64, align 4
  %p65 = getelementptr inbounds float, ptr addrspace(1) %in, i64 65
  %v65 = load volatile float, ptr addrspace(1) %p65, align 4
  %p66 = getelementptr inbounds float, ptr addrspace(1) %in, i64 66
  %v66 = load volatile float, ptr addrspace(1) %p66, align 4
  %p67 = getelementptr inbounds float, ptr addrspace(1) %in, i64 67
  %v67 = load volatile float, ptr addrspace(1) %p67, align 4
  %p68 = getelementptr inbounds float, ptr addrspace(1) %in, i64 68
  %v68 = load volatile float, ptr addrspace(1) %p68, align 4
  %p69 = getelementptr inbounds float, ptr addrspace(1) %in, i64 69
  %v69 = load volatile float, ptr addrspace(1) %p69, align 4
  %p70 = getelementptr inbounds float, ptr addrspace(1) %in, i64 70
  %v70 = load volatile float, ptr addrspace(1) %p70, align 4
  %p71 = getelementptr inbounds float, ptr addrspace(1) %in, i64 71
  %v71 = load volatile float, ptr addrspace(1) %p71, align 4
  call void asm sideeffect "", "v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v"(float %v0, float %v1, float %v2, float %v3, float %v4, float %v5, float %v6, float %v7, float %v8, float %v9, float %v10, float %v11, float %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18, float %v19, float %v20, float %v21, float %v22, float %v23, float %v24, float %v25, float %v26, float %v27, float %v28, float %v29, float %v30, float %v31, float %v32, float %v33, float %v34, float %v35, float %v36, float %v37, float %v38, float %v39, float %v40, float %v41, float %v42, float %v43, float %v44, float %v45, float %v46, float %v47, float %v48, float %v49, float %v50, float %v51, float %v52, float %v53, float %v54, float %v55, float %v56, float %v57, float %v58, float %v59, float %v60, float %v61, float %v62, float %v63, float %v64, float %v65, float %v66, float %v67, float %v68, float %v69, float %v70, float %v71)
  store float %v0, ptr addrspace(1) %out, align 4
  ret void
}

define amdgpu_kernel void @kernel_small(ptr addrspace(1) %in,
                                  ptr addrspace(1) %out) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %slot = and i32 %tid, 15
  %lp = getelementptr inbounds [16 x i32], ptr addrspace(3) @lds.small, i32 0, i32 %slot
  store volatile i32 %tid, ptr addrspace(3) %lp, align 4
  %unused = load volatile i32, ptr addrspace(3) %lp, align 4
  call void @create_pressure(ptr addrspace(1) %in, ptr addrspace(1) %out)
  ret void
}

define amdgpu_kernel void @kernel_large(ptr addrspace(1) %in,
                                  ptr addrspace(1) %out) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %slot = and i32 %tid, 15
  %lp = getelementptr inbounds [16384 x i32], ptr addrspace(3) @lds.large, i32 0, i32 %slot
  store volatile i32 %tid, ptr addrspace(3) %lp, align 4
  %unused = load volatile i32, ptr addrspace(3) %lp, align 4
  call void @create_pressure(ptr addrspace(1) %in, ptr addrspace(1) %out)
  ret void
}

attributes #0 = { noinline nounwind "amdgpu-flat-work-group-size"="64,64" }
