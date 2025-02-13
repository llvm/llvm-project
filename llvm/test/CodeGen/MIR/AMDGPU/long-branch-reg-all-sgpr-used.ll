; RUN: llc -mtriple=amdgcn -verify-machineinstrs -amdgpu-s-branch-bits=5 -stop-after=branch-relaxation  %s -o - | FileCheck %s

; Test long branch reserved register pass when all
; SGPRs are used

; CHECK-LABEL: {{^}}name: long_branch_used_all_sgprs
; CHECK: machineFunctionInfo:
; CHECK-NEXT:   explicitKernArgSize: 12
; CHECK-NEXT:   maxKernArgAlign: 8
; CHECK-NEXT:   ldsSize:         0
; CHECK-NEXT:   gdsSize:         0
; CHECK-NEXT:   dynLDSAlign:     1
; CHECK-NEXT:   isEntryFunction: true
; CHECK-NEXT:   isChainFunction: false
; CHECK-NEXT:   noSignedZerosFPMath: false
; CHECK-NEXT:   memoryBound:     false
; CHECK-NEXT:   waveLimiter:     false
; CHECK-NEXT:   hasSpilledSGPRs: false
; CHECK-NEXT:   hasSpilledVGPRs: false
; CHECK-NEXT:   scratchRSrcReg:  '$sgpr96_sgpr97_sgpr98_sgpr99'
; CHECK-NEXT:   frameOffsetReg:  '$fp_reg'
; CHECK-NEXT:   stackPtrOffsetReg: '$sgpr32'
; CHECK-NEXT:   bytesInStackArgArea: 0
; CHECK-NEXT:   returnsVoid:     true
; CHECK-NEXT:   argumentInfo:
; CHECK-NEXT:     kernargSegmentPtr: { reg: '$sgpr0_sgpr1' }
; CHECK-NEXT:     workGroupIDX:    { reg: '$sgpr2' }
; CHECK-NEXT:     privateSegmentWaveByteOffset: { reg: '$sgpr3' }
; CHECK-NEXT:     workItemIDX:     { reg: '$vgpr0' }
; CHECK-NEXT:   psInputAddr:     0
; CHECK-NEXT:   psInputEnable:   0
; CHECK-NEXT:   maxMemoryClusterDWords: 8
; CHECK-NEXT:   mode:
; CHECK-NEXT:     ieee:            true
; CHECK-NEXT:     dx10-clamp:      true
; CHECK-NEXT:     fp32-input-denormals: true
; CHECK-NEXT:     fp32-output-denormals: true
; CHECK-NEXT:     fp64-fp16-input-denormals: true
; CHECK-NEXT:     fp64-fp16-output-denormals: true
; CHECK-NEXT:   highBitsOf32BitAddress: 0
; CHECK-NEXT:   occupancy:       5
; CHECK-NEXT:   scavengeFI:      '%stack.0'
; CHECK-NEXT:   vgprForAGPRCopy: ''
; CHECK-NEXT:   sgprForEXECCopy: '$sgpr100_sgpr101'
; CHECK-NEXT:   longBranchReservedReg: ''
; CHECK-NEXT:   hasInitWholeWave: false
; CHECK-NEXT: body:
  define amdgpu_kernel void @long_branch_used_all_sgprs(ptr addrspace(1) %arg, i32 %cnd) #0 {
  entry:
    %long_branch_used_all_sgprs.kernarg.segment = call nonnull align 16 dereferenceable(48) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %cnd.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %long_branch_used_all_sgprs.kernarg.segment, i64 44, !amdgpu.uniform !0
    %cnd.load = load i32, ptr addrspace(4) %cnd.kernarg.offset, align 4, !invariant.load !0
    %sgpr0 = tail call i32 asm sideeffect "s_mov_b32 s0, 0", "={s0}"() #1
    %sgpr1 = tail call i32 asm sideeffect "s_mov_b32 s1, 0", "={s1}"() #1
    %sgpr2 = tail call i32 asm sideeffect "s_mov_b32 s2, 0", "={s2}"() #1
    %sgpr3 = tail call i32 asm sideeffect "s_mov_b32 s3, 0", "={s3}"() #1
    %sgpr4 = tail call i32 asm sideeffect "s_mov_b32 s4, 0", "={s4}"() #1
    %sgpr5 = tail call i32 asm sideeffect "s_mov_b32 s5, 0", "={s5}"() #1
    %sgpr6 = tail call i32 asm sideeffect "s_mov_b32 s6, 0", "={s6}"() #1
    %sgpr7 = tail call i32 asm sideeffect "s_mov_b32 s7, 0", "={s7}"() #1
    %sgpr8 = tail call i32 asm sideeffect "s_mov_b32 s8, 0", "={s8}"() #1
    %sgpr9 = tail call i32 asm sideeffect "s_mov_b32 s9, 0", "={s9}"() #1
    %sgpr10 = tail call i32 asm sideeffect "s_mov_b32 s10, 0", "={s10}"() #1
    %sgpr11 = tail call i32 asm sideeffect "s_mov_b32 s11, 0", "={s11}"() #1
    %sgpr12 = tail call i32 asm sideeffect "s_mov_b32 s12, 0", "={s12}"() #1
    %sgpr13 = tail call i32 asm sideeffect "s_mov_b32 s13, 0", "={s13}"() #1
    %sgpr14 = tail call i32 asm sideeffect "s_mov_b32 s14, 0", "={s14}"() #1
    %sgpr15 = tail call i32 asm sideeffect "s_mov_b32 s15, 0", "={s15}"() #1
    %sgpr16 = tail call i32 asm sideeffect "s_mov_b32 s16, 0", "={s16}"() #1
    %sgpr17 = tail call i32 asm sideeffect "s_mov_b32 s17, 0", "={s17}"() #1
    %sgpr18 = tail call i32 asm sideeffect "s_mov_b32 s18, 0", "={s18}"() #1
    %sgpr19 = tail call i32 asm sideeffect "s_mov_b32 s19, 0", "={s19}"() #1
    %sgpr20 = tail call i32 asm sideeffect "s_mov_b32 s20, 0", "={s20}"() #1
    %sgpr21 = tail call i32 asm sideeffect "s_mov_b32 s21, 0", "={s21}"() #1
    %sgpr22 = tail call i32 asm sideeffect "s_mov_b32 s22, 0", "={s22}"() #1
    %sgpr23 = tail call i32 asm sideeffect "s_mov_b32 s23, 0", "={s23}"() #1
    %sgpr24 = tail call i32 asm sideeffect "s_mov_b32 s24, 0", "={s24}"() #1
    %sgpr25 = tail call i32 asm sideeffect "s_mov_b32 s25, 0", "={s25}"() #1
    %sgpr26 = tail call i32 asm sideeffect "s_mov_b32 s26, 0", "={s26}"() #1
    %sgpr27 = tail call i32 asm sideeffect "s_mov_b32 s27, 0", "={s27}"() #1
    %sgpr28 = tail call i32 asm sideeffect "s_mov_b32 s28, 0", "={s28}"() #1
    %sgpr29 = tail call i32 asm sideeffect "s_mov_b32 s29, 0", "={s29}"() #1
    %sgpr30 = tail call i32 asm sideeffect "s_mov_b32 s30, 0", "={s30}"() #1
    %sgpr31 = tail call i32 asm sideeffect "s_mov_b32 s31, 0", "={s31}"() #1
    %sgpr32 = tail call i32 asm sideeffect "s_mov_b32 s32, 0", "={s32}"() #1
    %sgpr33 = tail call i32 asm sideeffect "s_mov_b32 s33, 0", "={s33}"() #1
    %sgpr34 = tail call i32 asm sideeffect "s_mov_b32 s34, 0", "={s34}"() #1
    %sgpr35 = tail call i32 asm sideeffect "s_mov_b32 s35, 0", "={s35}"() #1
    %sgpr36 = tail call i32 asm sideeffect "s_mov_b32 s36, 0", "={s36}"() #1
    %sgpr37 = tail call i32 asm sideeffect "s_mov_b32 s37, 0", "={s37}"() #1
    %sgpr38 = tail call i32 asm sideeffect "s_mov_b32 s38, 0", "={s38}"() #1
    %sgpr39 = tail call i32 asm sideeffect "s_mov_b32 s39, 0", "={s39}"() #1
    %sgpr40 = tail call i32 asm sideeffect "s_mov_b32 s40, 0", "={s40}"() #1
    %sgpr41 = tail call i32 asm sideeffect "s_mov_b32 s41, 0", "={s41}"() #1
    %sgpr42 = tail call i32 asm sideeffect "s_mov_b32 s42, 0", "={s42}"() #1
    %sgpr43 = tail call i32 asm sideeffect "s_mov_b32 s43, 0", "={s43}"() #1
    %sgpr44 = tail call i32 asm sideeffect "s_mov_b32 s44, 0", "={s44}"() #1
    %sgpr45 = tail call i32 asm sideeffect "s_mov_b32 s45, 0", "={s45}"() #1
    %sgpr46 = tail call i32 asm sideeffect "s_mov_b32 s46, 0", "={s46}"() #1
    %sgpr47 = tail call i32 asm sideeffect "s_mov_b32 s47, 0", "={s47}"() #1
    %sgpr48 = tail call i32 asm sideeffect "s_mov_b32 s48, 0", "={s48}"() #1
    %sgpr49 = tail call i32 asm sideeffect "s_mov_b32 s49, 0", "={s49}"() #1
    %sgpr50 = tail call i32 asm sideeffect "s_mov_b32 s50, 0", "={s50}"() #1
    %sgpr51 = tail call i32 asm sideeffect "s_mov_b32 s51, 0", "={s51}"() #1
    %sgpr52 = tail call i32 asm sideeffect "s_mov_b32 s52, 0", "={s52}"() #1
    %sgpr53 = tail call i32 asm sideeffect "s_mov_b32 s53, 0", "={s53}"() #1
    %sgpr54 = tail call i32 asm sideeffect "s_mov_b32 s54, 0", "={s54}"() #1
    %sgpr55 = tail call i32 asm sideeffect "s_mov_b32 s55, 0", "={s55}"() #1
    %sgpr56 = tail call i32 asm sideeffect "s_mov_b32 s56, 0", "={s56}"() #1
    %sgpr57 = tail call i32 asm sideeffect "s_mov_b32 s57, 0", "={s57}"() #1
    %sgpr58 = tail call i32 asm sideeffect "s_mov_b32 s58, 0", "={s58}"() #1
    %sgpr59 = tail call i32 asm sideeffect "s_mov_b32 s59, 0", "={s59}"() #1
    %sgpr60 = tail call i32 asm sideeffect "s_mov_b32 s60, 0", "={s60}"() #1
    %sgpr61 = tail call i32 asm sideeffect "s_mov_b32 s61, 0", "={s61}"() #1
    %sgpr62 = tail call i32 asm sideeffect "s_mov_b32 s62, 0", "={s62}"() #1
    %sgpr63 = tail call i32 asm sideeffect "s_mov_b32 s63, 0", "={s63}"() #1
    %sgpr64 = tail call i32 asm sideeffect "s_mov_b32 s64, 0", "={s64}"() #1
    %sgpr65 = tail call i32 asm sideeffect "s_mov_b32 s65, 0", "={s65}"() #1
    %sgpr66 = tail call i32 asm sideeffect "s_mov_b32 s66, 0", "={s66}"() #1
    %sgpr67 = tail call i32 asm sideeffect "s_mov_b32 s67, 0", "={s67}"() #1
    %sgpr68 = tail call i32 asm sideeffect "s_mov_b32 s68, 0", "={s68}"() #1
    %sgpr69 = tail call i32 asm sideeffect "s_mov_b32 s69, 0", "={s69}"() #1
    %sgpr70 = tail call i32 asm sideeffect "s_mov_b32 s70, 0", "={s70}"() #1
    %sgpr71 = tail call i32 asm sideeffect "s_mov_b32 s71, 0", "={s71}"() #1
    %sgpr72 = tail call i32 asm sideeffect "s_mov_b32 s72, 0", "={s72}"() #1
    %sgpr73 = tail call i32 asm sideeffect "s_mov_b32 s73, 0", "={s73}"() #1
    %sgpr74 = tail call i32 asm sideeffect "s_mov_b32 s74, 0", "={s74}"() #1
    %sgpr75 = tail call i32 asm sideeffect "s_mov_b32 s75, 0", "={s75}"() #1
    %sgpr76 = tail call i32 asm sideeffect "s_mov_b32 s76, 0", "={s76}"() #1
    %sgpr77 = tail call i32 asm sideeffect "s_mov_b32 s77, 0", "={s77}"() #1
    %sgpr78 = tail call i32 asm sideeffect "s_mov_b32 s78, 0", "={s78}"() #1
    %sgpr79 = tail call i32 asm sideeffect "s_mov_b32 s79, 0", "={s79}"() #1
    %sgpr80 = tail call i32 asm sideeffect "s_mov_b32 s80, 0", "={s80}"() #1
    %sgpr81 = tail call i32 asm sideeffect "s_mov_b32 s81, 0", "={s81}"() #1
    %sgpr82 = tail call i32 asm sideeffect "s_mov_b32 s82, 0", "={s82}"() #1
    %sgpr83 = tail call i32 asm sideeffect "s_mov_b32 s83, 0", "={s83}"() #1
    %sgpr84 = tail call i32 asm sideeffect "s_mov_b32 s84, 0", "={s84}"() #1
    %sgpr85 = tail call i32 asm sideeffect "s_mov_b32 s85, 0", "={s85}"() #1
    %sgpr86 = tail call i32 asm sideeffect "s_mov_b32 s86, 0", "={s86}"() #1
    %sgpr87 = tail call i32 asm sideeffect "s_mov_b32 s87, 0", "={s87}"() #1
    %sgpr88 = tail call i32 asm sideeffect "s_mov_b32 s88, 0", "={s88}"() #1
    %sgpr89 = tail call i32 asm sideeffect "s_mov_b32 s89, 0", "={s89}"() #1
    %sgpr90 = tail call i32 asm sideeffect "s_mov_b32 s90, 0", "={s90}"() #1
    %sgpr91 = tail call i32 asm sideeffect "s_mov_b32 s91, 0", "={s91}"() #1
    %sgpr92 = tail call i32 asm sideeffect "s_mov_b32 s92, 0", "={s92}"() #1
    %sgpr93 = tail call i32 asm sideeffect "s_mov_b32 s93, 0", "={s93}"() #1
    %sgpr94 = tail call i32 asm sideeffect "s_mov_b32 s94, 0", "={s94}"() #1
    %sgpr95 = tail call i32 asm sideeffect "s_mov_b32 s95, 0", "={s95}"() #1
    %sgpr96 = tail call i32 asm sideeffect "s_mov_b32 s96, 0", "={s96}"() #1
    %sgpr97 = tail call i32 asm sideeffect "s_mov_b32 s97, 0", "={s97}"() #1
    %sgpr98 = tail call i32 asm sideeffect "s_mov_b32 s98, 0", "={s98}"() #1
    %sgpr99 = tail call i32 asm sideeffect "s_mov_b32 s99, 0", "={s99}"() #1
    %sgpr100 = tail call i32 asm sideeffect "s_mov_b32 s100, 0", "={s100}"() #1
    %sgpr101 = tail call i32 asm sideeffect "s_mov_b32 s101, 0", "={s101}"() #1
    %vcc_lo = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={vcc_lo}"() #1
    %vcc_hi = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={vcc_hi}"() #1
    %cmp = icmp ne i32 %cnd.load, 0
    br i1 %cmp, label %bb2, label %bb3, !amdgpu.uniform !0

  bb2:                                              ; preds = %entry
    call void asm sideeffect "v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64", ""() #1
    br label %bb3, !amdgpu.uniform !0

  bb3:                                              ; preds = %bb2, %entry
    tail call void asm sideeffect "; reg use $0", "{s0}"(i32 %sgpr0) #1
    tail call void asm sideeffect "; reg use $0", "{s1}"(i32 %sgpr1) #1
    tail call void asm sideeffect "; reg use $0", "{s2}"(i32 %sgpr2) #1
    tail call void asm sideeffect "; reg use $0", "{s3}"(i32 %sgpr3) #1
    tail call void asm sideeffect "; reg use $0", "{s4}"(i32 %sgpr4) #1
    tail call void asm sideeffect "; reg use $0", "{s5}"(i32 %sgpr5) #1
    tail call void asm sideeffect "; reg use $0", "{s6}"(i32 %sgpr6) #1
    tail call void asm sideeffect "; reg use $0", "{s7}"(i32 %sgpr7) #1
    tail call void asm sideeffect "; reg use $0", "{s8}"(i32 %sgpr8) #1
    tail call void asm sideeffect "; reg use $0", "{s9}"(i32 %sgpr9) #1
    tail call void asm sideeffect "; reg use $0", "{s10}"(i32 %sgpr10) #1
    tail call void asm sideeffect "; reg use $0", "{s11}"(i32 %sgpr11) #1
    tail call void asm sideeffect "; reg use $0", "{s12}"(i32 %sgpr12) #1
    tail call void asm sideeffect "; reg use $0", "{s13}"(i32 %sgpr13) #1
    tail call void asm sideeffect "; reg use $0", "{s14}"(i32 %sgpr14) #1
    tail call void asm sideeffect "; reg use $0", "{s15}"(i32 %sgpr15) #1
    tail call void asm sideeffect "; reg use $0", "{s16}"(i32 %sgpr16) #1
    tail call void asm sideeffect "; reg use $0", "{s17}"(i32 %sgpr17) #1
    tail call void asm sideeffect "; reg use $0", "{s18}"(i32 %sgpr18) #1
    tail call void asm sideeffect "; reg use $0", "{s19}"(i32 %sgpr19) #1
    tail call void asm sideeffect "; reg use $0", "{s20}"(i32 %sgpr20) #1
    tail call void asm sideeffect "; reg use $0", "{s21}"(i32 %sgpr21) #1
    tail call void asm sideeffect "; reg use $0", "{s22}"(i32 %sgpr22) #1
    tail call void asm sideeffect "; reg use $0", "{s23}"(i32 %sgpr23) #1
    tail call void asm sideeffect "; reg use $0", "{s24}"(i32 %sgpr24) #1
    tail call void asm sideeffect "; reg use $0", "{s25}"(i32 %sgpr25) #1
    tail call void asm sideeffect "; reg use $0", "{s26}"(i32 %sgpr26) #1
    tail call void asm sideeffect "; reg use $0", "{s27}"(i32 %sgpr27) #1
    tail call void asm sideeffect "; reg use $0", "{s28}"(i32 %sgpr28) #1
    tail call void asm sideeffect "; reg use $0", "{s29}"(i32 %sgpr29) #1
    tail call void asm sideeffect "; reg use $0", "{s30}"(i32 %sgpr30) #1
    tail call void asm sideeffect "; reg use $0", "{s31}"(i32 %sgpr31) #1
    tail call void asm sideeffect "; reg use $0", "{s32}"(i32 %sgpr32) #1
    tail call void asm sideeffect "; reg use $0", "{s33}"(i32 %sgpr33) #1
    tail call void asm sideeffect "; reg use $0", "{s34}"(i32 %sgpr34) #1
    tail call void asm sideeffect "; reg use $0", "{s35}"(i32 %sgpr35) #1
    tail call void asm sideeffect "; reg use $0", "{s36}"(i32 %sgpr36) #1
    tail call void asm sideeffect "; reg use $0", "{s37}"(i32 %sgpr37) #1
    tail call void asm sideeffect "; reg use $0", "{s38}"(i32 %sgpr38) #1
    tail call void asm sideeffect "; reg use $0", "{s39}"(i32 %sgpr39) #1
    tail call void asm sideeffect "; reg use $0", "{s40}"(i32 %sgpr40) #1
    tail call void asm sideeffect "; reg use $0", "{s41}"(i32 %sgpr41) #1
    tail call void asm sideeffect "; reg use $0", "{s42}"(i32 %sgpr42) #1
    tail call void asm sideeffect "; reg use $0", "{s43}"(i32 %sgpr43) #1
    tail call void asm sideeffect "; reg use $0", "{s44}"(i32 %sgpr44) #1
    tail call void asm sideeffect "; reg use $0", "{s45}"(i32 %sgpr45) #1
    tail call void asm sideeffect "; reg use $0", "{s46}"(i32 %sgpr46) #1
    tail call void asm sideeffect "; reg use $0", "{s47}"(i32 %sgpr47) #1
    tail call void asm sideeffect "; reg use $0", "{s48}"(i32 %sgpr48) #1
    tail call void asm sideeffect "; reg use $0", "{s49}"(i32 %sgpr49) #1
    tail call void asm sideeffect "; reg use $0", "{s50}"(i32 %sgpr50) #1
    tail call void asm sideeffect "; reg use $0", "{s51}"(i32 %sgpr51) #1
    tail call void asm sideeffect "; reg use $0", "{s52}"(i32 %sgpr52) #1
    tail call void asm sideeffect "; reg use $0", "{s53}"(i32 %sgpr53) #1
    tail call void asm sideeffect "; reg use $0", "{s54}"(i32 %sgpr54) #1
    tail call void asm sideeffect "; reg use $0", "{s55}"(i32 %sgpr55) #1
    tail call void asm sideeffect "; reg use $0", "{s56}"(i32 %sgpr56) #1
    tail call void asm sideeffect "; reg use $0", "{s57}"(i32 %sgpr57) #1
    tail call void asm sideeffect "; reg use $0", "{s58}"(i32 %sgpr58) #1
    tail call void asm sideeffect "; reg use $0", "{s59}"(i32 %sgpr59) #1
    tail call void asm sideeffect "; reg use $0", "{s60}"(i32 %sgpr60) #1
    tail call void asm sideeffect "; reg use $0", "{s61}"(i32 %sgpr61) #1
    tail call void asm sideeffect "; reg use $0", "{s62}"(i32 %sgpr62) #1
    tail call void asm sideeffect "; reg use $0", "{s63}"(i32 %sgpr63) #1
    tail call void asm sideeffect "; reg use $0", "{s64}"(i32 %sgpr64) #1
    tail call void asm sideeffect "; reg use $0", "{s65}"(i32 %sgpr65) #1
    tail call void asm sideeffect "; reg use $0", "{s66}"(i32 %sgpr66) #1
    tail call void asm sideeffect "; reg use $0", "{s67}"(i32 %sgpr67) #1
    tail call void asm sideeffect "; reg use $0", "{s68}"(i32 %sgpr68) #1
    tail call void asm sideeffect "; reg use $0", "{s69}"(i32 %sgpr69) #1
    tail call void asm sideeffect "; reg use $0", "{s70}"(i32 %sgpr70) #1
    tail call void asm sideeffect "; reg use $0", "{s71}"(i32 %sgpr71) #1
    tail call void asm sideeffect "; reg use $0", "{s72}"(i32 %sgpr72) #1
    tail call void asm sideeffect "; reg use $0", "{s73}"(i32 %sgpr73) #1
    tail call void asm sideeffect "; reg use $0", "{s74}"(i32 %sgpr74) #1
    tail call void asm sideeffect "; reg use $0", "{s75}"(i32 %sgpr75) #1
    tail call void asm sideeffect "; reg use $0", "{s76}"(i32 %sgpr76) #1
    tail call void asm sideeffect "; reg use $0", "{s77}"(i32 %sgpr77) #1
    tail call void asm sideeffect "; reg use $0", "{s78}"(i32 %sgpr78) #1
    tail call void asm sideeffect "; reg use $0", "{s79}"(i32 %sgpr79) #1
    tail call void asm sideeffect "; reg use $0", "{s80}"(i32 %sgpr80) #1
    tail call void asm sideeffect "; reg use $0", "{s81}"(i32 %sgpr81) #1
    tail call void asm sideeffect "; reg use $0", "{s82}"(i32 %sgpr82) #1
    tail call void asm sideeffect "; reg use $0", "{s83}"(i32 %sgpr83) #1
    tail call void asm sideeffect "; reg use $0", "{s84}"(i32 %sgpr84) #1
    tail call void asm sideeffect "; reg use $0", "{s85}"(i32 %sgpr85) #1
    tail call void asm sideeffect "; reg use $0", "{s86}"(i32 %sgpr86) #1
    tail call void asm sideeffect "; reg use $0", "{s87}"(i32 %sgpr87) #1
    tail call void asm sideeffect "; reg use $0", "{s88}"(i32 %sgpr88) #1
    tail call void asm sideeffect "; reg use $0", "{s89}"(i32 %sgpr89) #1
    tail call void asm sideeffect "; reg use $0", "{s90}"(i32 %sgpr90) #1
    tail call void asm sideeffect "; reg use $0", "{s91}"(i32 %sgpr91) #1
    tail call void asm sideeffect "; reg use $0", "{s92}"(i32 %sgpr92) #1
    tail call void asm sideeffect "; reg use $0", "{s93}"(i32 %sgpr93) #1
    tail call void asm sideeffect "; reg use $0", "{s94}"(i32 %sgpr94) #1
    tail call void asm sideeffect "; reg use $0", "{s95}"(i32 %sgpr95) #1
    tail call void asm sideeffect "; reg use $0", "{s96}"(i32 %sgpr96) #1
    tail call void asm sideeffect "; reg use $0", "{s97}"(i32 %sgpr97) #1
    tail call void asm sideeffect "; reg use $0", "{s98}"(i32 %sgpr98) #1
    tail call void asm sideeffect "; reg use $0", "{s99}"(i32 %sgpr99) #1
    tail call void asm sideeffect "; reg use $0", "{s100}"(i32 %sgpr100) #1
    tail call void asm sideeffect "; reg use $0", "{s101}"(i32 %sgpr101) #1
    tail call void asm sideeffect "; reg use $0", "{vcc_lo}"(i32 %vcc_lo) #1
    tail call void asm sideeffect "; reg use $0", "{vcc_hi}"(i32 %vcc_hi) #1
    ret void
  }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
; CHECK-LABEL: {{^}}name: long_branch_high_num_sgprs_used
; CHECK: machineFunctionInfo:
; CHECK-NEXT:   explicitKernArgSize: 12
; CHECK-NEXT:   maxKernArgAlign: 8
; CHECK-NEXT:   ldsSize:         0
; CHECK-NEXT:   gdsSize:         0
; CHECK-NEXT:   dynLDSAlign:     1
; CHECK-NEXT:   isEntryFunction: true
; CHECK-NEXT:   isChainFunction: false
; CHECK-NEXT:   noSignedZerosFPMath: false
; CHECK-NEXT:   memoryBound:     false
; CHECK-NEXT:   waveLimiter:     false
; CHECK-NEXT:   hasSpilledSGPRs: false
; CHECK-NEXT:   hasSpilledVGPRs: false
; CHECK-NEXT:   scratchRSrcReg:  '$sgpr96_sgpr97_sgpr98_sgpr99'
; CHECK-NEXT:   frameOffsetReg:  '$fp_reg'
; CHECK-NEXT:   stackPtrOffsetReg: '$sgpr32'
; CHECK-NEXT:   bytesInStackArgArea: 0
; CHECK-NEXT:   returnsVoid:     true
; CHECK-NEXT:   argumentInfo:
; CHECK-NEXT:     kernargSegmentPtr: { reg: '$sgpr0_sgpr1' }
; CHECK-NEXT:     workGroupIDX:    { reg: '$sgpr2' }
; CHECK-NEXT:     privateSegmentWaveByteOffset: { reg: '$sgpr3' }
; CHECK-NEXT:     workItemIDX:     { reg: '$vgpr0' }
; CHECK-NEXT:   psInputAddr:     0
; CHECK-NEXT:   psInputEnable:   0
; CHECK-NEXT:   maxMemoryClusterDWords: 8
; CHECK-NEXT:   mode:
; CHECK-NEXT:     ieee:            true
; CHECK-NEXT:     dx10-clamp:      true
; CHECK-NEXT:     fp32-input-denormals: true
; CHECK-NEXT:     fp32-output-denormals: true
; CHECK-NEXT:     fp64-fp16-input-denormals: true
; CHECK-NEXT:     fp64-fp16-output-denormals: true
; CHECK-NEXT:   highBitsOf32BitAddress: 0
; CHECK-NEXT:   occupancy:       5
; CHECK-NEXT:   scavengeFI:      '%stack.0'
; CHECK-NEXT:   vgprForAGPRCopy: ''
; CHECK-NEXT:   sgprForEXECCopy: '$sgpr100_sgpr101'
; CHECK-NEXT:   longBranchReservedReg: ''
; CHECK-NEXT:   hasInitWholeWave: false
; CHECK-NEXT: body:
  define amdgpu_kernel void @long_branch_high_num_sgprs_used(ptr addrspace(1) %arg, i32 %cnd) #0 {
  entry:
    %long_branch_used_all_sgprs.kernarg.segment = call nonnull align 16 dereferenceable(48) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
    %cnd.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %long_branch_used_all_sgprs.kernarg.segment, i64 44, !amdgpu.uniform !0
    %cnd.load = load i32, ptr addrspace(4) %cnd.kernarg.offset, align 4, !invariant.load !0
    %sgpr0 = tail call i32 asm sideeffect "s_mov_b32 s0, 0", "={s0}"() #1
    %sgpr1 = tail call i32 asm sideeffect "s_mov_b32 s1, 0", "={s1}"() #1
    %sgpr2 = tail call i32 asm sideeffect "s_mov_b32 s2, 0", "={s2}"() #1
    %sgpr3 = tail call i32 asm sideeffect "s_mov_b32 s3, 0", "={s3}"() #1
    %sgpr4 = tail call i32 asm sideeffect "s_mov_b32 s4, 0", "={s4}"() #1
    %sgpr5 = tail call i32 asm sideeffect "s_mov_b32 s5, 0", "={s5}"() #1
    %sgpr6 = tail call i32 asm sideeffect "s_mov_b32 s6, 0", "={s6}"() #1
    %sgpr7 = tail call i32 asm sideeffect "s_mov_b32 s7, 0", "={s7}"() #1
    %sgpr8 = tail call i32 asm sideeffect "s_mov_b32 s8, 0", "={s8}"() #1
    %sgpr9 = tail call i32 asm sideeffect "s_mov_b32 s9, 0", "={s9}"() #1
    %sgpr10 = tail call i32 asm sideeffect "s_mov_b32 s10, 0", "={s10}"() #1
    %sgpr11 = tail call i32 asm sideeffect "s_mov_b32 s11, 0", "={s11}"() #1
    %sgpr12 = tail call i32 asm sideeffect "s_mov_b32 s12, 0", "={s12}"() #1
    %sgpr13 = tail call i32 asm sideeffect "s_mov_b32 s13, 0", "={s13}"() #1
    %sgpr14 = tail call i32 asm sideeffect "s_mov_b32 s14, 0", "={s14}"() #1
    %sgpr15 = tail call i32 asm sideeffect "s_mov_b32 s15, 0", "={s15}"() #1
    %sgpr16 = tail call i32 asm sideeffect "s_mov_b32 s16, 0", "={s16}"() #1
    %sgpr17 = tail call i32 asm sideeffect "s_mov_b32 s17, 0", "={s17}"() #1
    %sgpr18 = tail call i32 asm sideeffect "s_mov_b32 s18, 0", "={s18}"() #1
    %sgpr19 = tail call i32 asm sideeffect "s_mov_b32 s19, 0", "={s19}"() #1
    %sgpr20 = tail call i32 asm sideeffect "s_mov_b32 s20, 0", "={s20}"() #1
    %sgpr21 = tail call i32 asm sideeffect "s_mov_b32 s21, 0", "={s21}"() #1
    %sgpr22 = tail call i32 asm sideeffect "s_mov_b32 s22, 0", "={s22}"() #1
    %sgpr23 = tail call i32 asm sideeffect "s_mov_b32 s23, 0", "={s23}"() #1
    %sgpr24 = tail call i32 asm sideeffect "s_mov_b32 s24, 0", "={s24}"() #1
    %sgpr25 = tail call i32 asm sideeffect "s_mov_b32 s25, 0", "={s25}"() #1
    %sgpr26 = tail call i32 asm sideeffect "s_mov_b32 s26, 0", "={s26}"() #1
    %sgpr27 = tail call i32 asm sideeffect "s_mov_b32 s27, 0", "={s27}"() #1
    %sgpr28 = tail call i32 asm sideeffect "s_mov_b32 s28, 0", "={s28}"() #1
    %sgpr29 = tail call i32 asm sideeffect "s_mov_b32 s29, 0", "={s29}"() #1
    %sgpr30 = tail call i32 asm sideeffect "s_mov_b32 s30, 0", "={s30}"() #1
    %sgpr31 = tail call i32 asm sideeffect "s_mov_b32 s31, 0", "={s31}"() #1
    %sgpr32 = tail call i32 asm sideeffect "s_mov_b32 s32, 0", "={s32}"() #1
    %sgpr33 = tail call i32 asm sideeffect "s_mov_b32 s33, 0", "={s33}"() #1
    %sgpr34 = tail call i32 asm sideeffect "s_mov_b32 s34, 0", "={s34}"() #1
    %sgpr35 = tail call i32 asm sideeffect "s_mov_b32 s35, 0", "={s35}"() #1
    %sgpr36 = tail call i32 asm sideeffect "s_mov_b32 s36, 0", "={s36}"() #1
    %sgpr37 = tail call i32 asm sideeffect "s_mov_b32 s37, 0", "={s37}"() #1
    %sgpr38 = tail call i32 asm sideeffect "s_mov_b32 s38, 0", "={s38}"() #1
    %sgpr39 = tail call i32 asm sideeffect "s_mov_b32 s39, 0", "={s39}"() #1
    %sgpr40 = tail call i32 asm sideeffect "s_mov_b32 s40, 0", "={s40}"() #1
    %sgpr41 = tail call i32 asm sideeffect "s_mov_b32 s41, 0", "={s41}"() #1
    %sgpr42 = tail call i32 asm sideeffect "s_mov_b32 s42, 0", "={s42}"() #1
    %sgpr43 = tail call i32 asm sideeffect "s_mov_b32 s43, 0", "={s43}"() #1
    %sgpr44 = tail call i32 asm sideeffect "s_mov_b32 s44, 0", "={s44}"() #1
    %sgpr45 = tail call i32 asm sideeffect "s_mov_b32 s45, 0", "={s45}"() #1
    %sgpr46 = tail call i32 asm sideeffect "s_mov_b32 s46, 0", "={s46}"() #1
    %sgpr47 = tail call i32 asm sideeffect "s_mov_b32 s47, 0", "={s47}"() #1
    %sgpr48 = tail call i32 asm sideeffect "s_mov_b32 s48, 0", "={s48}"() #1
    %sgpr49 = tail call i32 asm sideeffect "s_mov_b32 s49, 0", "={s49}"() #1
    %sgpr50 = tail call i32 asm sideeffect "s_mov_b32 s50, 0", "={s50}"() #1
    %sgpr51 = tail call i32 asm sideeffect "s_mov_b32 s51, 0", "={s51}"() #1
    %sgpr52 = tail call i32 asm sideeffect "s_mov_b32 s52, 0", "={s52}"() #1
    %sgpr53 = tail call i32 asm sideeffect "s_mov_b32 s53, 0", "={s53}"() #1
    %sgpr54 = tail call i32 asm sideeffect "s_mov_b32 s54, 0", "={s54}"() #1
    %sgpr55 = tail call i32 asm sideeffect "s_mov_b32 s55, 0", "={s55}"() #1
    %sgpr56 = tail call i32 asm sideeffect "s_mov_b32 s56, 0", "={s56}"() #1
    %sgpr57 = tail call i32 asm sideeffect "s_mov_b32 s57, 0", "={s57}"() #1
    %sgpr58 = tail call i32 asm sideeffect "s_mov_b32 s58, 0", "={s58}"() #1
    %sgpr59 = tail call i32 asm sideeffect "s_mov_b32 s59, 0", "={s59}"() #1
    %sgpr60 = tail call i32 asm sideeffect "s_mov_b32 s60, 0", "={s60}"() #1
    %sgpr61 = tail call i32 asm sideeffect "s_mov_b32 s61, 0", "={s61}"() #1
    %sgpr62 = tail call i32 asm sideeffect "s_mov_b32 s62, 0", "={s62}"() #1
    %sgpr63 = tail call i32 asm sideeffect "s_mov_b32 s63, 0", "={s63}"() #1
    %sgpr64 = tail call i32 asm sideeffect "s_mov_b32 s64, 0", "={s64}"() #1
    %sgpr65 = tail call i32 asm sideeffect "s_mov_b32 s65, 0", "={s65}"() #1
    %sgpr66 = tail call i32 asm sideeffect "s_mov_b32 s66, 0", "={s66}"() #1
    %sgpr67 = tail call i32 asm sideeffect "s_mov_b32 s67, 0", "={s67}"() #1
    %sgpr68 = tail call i32 asm sideeffect "s_mov_b32 s68, 0", "={s68}"() #1
    %sgpr69 = tail call i32 asm sideeffect "s_mov_b32 s69, 0", "={s69}"() #1
    %sgpr70 = tail call i32 asm sideeffect "s_mov_b32 s70, 0", "={s70}"() #1
    %sgpr71 = tail call i32 asm sideeffect "s_mov_b32 s71, 0", "={s71}"() #1
    %sgpr72 = tail call i32 asm sideeffect "s_mov_b32 s72, 0", "={s72}"() #1
    %sgpr73 = tail call i32 asm sideeffect "s_mov_b32 s73, 0", "={s73}"() #1
    %sgpr74 = tail call i32 asm sideeffect "s_mov_b32 s74, 0", "={s74}"() #1
    %sgpr75 = tail call i32 asm sideeffect "s_mov_b32 s75, 0", "={s75}"() #1
    %sgpr76 = tail call i32 asm sideeffect "s_mov_b32 s76, 0", "={s76}"() #1
    %sgpr77 = tail call i32 asm sideeffect "s_mov_b32 s77, 0", "={s77}"() #1
    %sgpr78 = tail call i32 asm sideeffect "s_mov_b32 s78, 0", "={s78}"() #1
    %sgpr79 = tail call i32 asm sideeffect "s_mov_b32 s79, 0", "={s79}"() #1
    %sgpr80 = tail call i32 asm sideeffect "s_mov_b32 s80, 0", "={s80}"() #1
    %sgpr81 = tail call i32 asm sideeffect "s_mov_b32 s81, 0", "={s81}"() #1
    %sgpr82 = tail call i32 asm sideeffect "s_mov_b32 s82, 0", "={s82}"() #1
    %sgpr83 = tail call i32 asm sideeffect "s_mov_b32 s83, 0", "={s83}"() #1
    %sgpr84 = tail call i32 asm sideeffect "s_mov_b32 s84, 0", "={s84}"() #1
    %sgpr85 = tail call i32 asm sideeffect "s_mov_b32 s85, 0", "={s85}"() #1
    %sgpr86 = tail call i32 asm sideeffect "s_mov_b32 s86, 0", "={s86}"() #1
    %sgpr87 = tail call i32 asm sideeffect "s_mov_b32 s87, 0", "={s87}"() #1
    %sgpr88 = tail call i32 asm sideeffect "s_mov_b32 s88, 0", "={s88}"() #1
    %sgpr89 = tail call i32 asm sideeffect "s_mov_b32 s89, 0", "={s89}"() #1
    %sgpr90 = tail call i32 asm sideeffect "s_mov_b32 s90, 0", "={s90}"() #1
    %sgpr91 = tail call i32 asm sideeffect "s_mov_b32 s91, 0", "={s91}"() #1
    %sgpr92 = tail call i32 asm sideeffect "s_mov_b32 s92, 0", "={s92}"() #1
    %sgpr93 = tail call i32 asm sideeffect "s_mov_b32 s93, 0", "={s93}"() #1
    %sgpr94 = tail call i32 asm sideeffect "s_mov_b32 s94, 0", "={s94}"() #1
    %sgpr95 = tail call i32 asm sideeffect "s_mov_b32 s95, 0", "={s95}"() #1
    %sgpr96 = tail call i32 asm sideeffect "s_mov_b32 s96, 0", "={s96}"() #1
    %sgpr97 = tail call i32 asm sideeffect "s_mov_b32 s97, 0", "={s97}"() #1
    %vcc_lo = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={vcc_lo}"() #1
    %vcc_hi = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={vcc_hi}"() #1
    %cmp = icmp ne i32 %cnd.load, 0
    br i1 %cmp, label %bb2, label %bb3, !amdgpu.uniform !0

  bb2:                                              ; preds = %entry
    call void asm sideeffect "v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64\0A    v_nop_e64", ""() #1
    br label %bb3, !amdgpu.uniform !0

  bb3:                                              ; preds = %bb2, %entry
    tail call void asm sideeffect "; reg use $0", "{s0}"(i32 %sgpr0) #1
    tail call void asm sideeffect "; reg use $0", "{s1}"(i32 %sgpr1) #1
    tail call void asm sideeffect "; reg use $0", "{s2}"(i32 %sgpr2) #1
    tail call void asm sideeffect "; reg use $0", "{s3}"(i32 %sgpr3) #1
    tail call void asm sideeffect "; reg use $0", "{s4}"(i32 %sgpr4) #1
    tail call void asm sideeffect "; reg use $0", "{s5}"(i32 %sgpr5) #1
    tail call void asm sideeffect "; reg use $0", "{s6}"(i32 %sgpr6) #1
    tail call void asm sideeffect "; reg use $0", "{s7}"(i32 %sgpr7) #1
    tail call void asm sideeffect "; reg use $0", "{s8}"(i32 %sgpr8) #1
    tail call void asm sideeffect "; reg use $0", "{s9}"(i32 %sgpr9) #1
    tail call void asm sideeffect "; reg use $0", "{s10}"(i32 %sgpr10) #1
    tail call void asm sideeffect "; reg use $0", "{s11}"(i32 %sgpr11) #1
    tail call void asm sideeffect "; reg use $0", "{s12}"(i32 %sgpr12) #1
    tail call void asm sideeffect "; reg use $0", "{s13}"(i32 %sgpr13) #1
    tail call void asm sideeffect "; reg use $0", "{s14}"(i32 %sgpr14) #1
    tail call void asm sideeffect "; reg use $0", "{s15}"(i32 %sgpr15) #1
    tail call void asm sideeffect "; reg use $0", "{s16}"(i32 %sgpr16) #1
    tail call void asm sideeffect "; reg use $0", "{s17}"(i32 %sgpr17) #1
    tail call void asm sideeffect "; reg use $0", "{s18}"(i32 %sgpr18) #1
    tail call void asm sideeffect "; reg use $0", "{s19}"(i32 %sgpr19) #1
    tail call void asm sideeffect "; reg use $0", "{s20}"(i32 %sgpr20) #1
    tail call void asm sideeffect "; reg use $0", "{s21}"(i32 %sgpr21) #1
    tail call void asm sideeffect "; reg use $0", "{s22}"(i32 %sgpr22) #1
    tail call void asm sideeffect "; reg use $0", "{s23}"(i32 %sgpr23) #1
    tail call void asm sideeffect "; reg use $0", "{s24}"(i32 %sgpr24) #1
    tail call void asm sideeffect "; reg use $0", "{s25}"(i32 %sgpr25) #1
    tail call void asm sideeffect "; reg use $0", "{s26}"(i32 %sgpr26) #1
    tail call void asm sideeffect "; reg use $0", "{s27}"(i32 %sgpr27) #1
    tail call void asm sideeffect "; reg use $0", "{s28}"(i32 %sgpr28) #1
    tail call void asm sideeffect "; reg use $0", "{s29}"(i32 %sgpr29) #1
    tail call void asm sideeffect "; reg use $0", "{s30}"(i32 %sgpr30) #1
    tail call void asm sideeffect "; reg use $0", "{s31}"(i32 %sgpr31) #1
    tail call void asm sideeffect "; reg use $0", "{s32}"(i32 %sgpr32) #1
    tail call void asm sideeffect "; reg use $0", "{s33}"(i32 %sgpr33) #1
    tail call void asm sideeffect "; reg use $0", "{s34}"(i32 %sgpr34) #1
    tail call void asm sideeffect "; reg use $0", "{s35}"(i32 %sgpr35) #1
    tail call void asm sideeffect "; reg use $0", "{s36}"(i32 %sgpr36) #1
    tail call void asm sideeffect "; reg use $0", "{s37}"(i32 %sgpr37) #1
    tail call void asm sideeffect "; reg use $0", "{s38}"(i32 %sgpr38) #1
    tail call void asm sideeffect "; reg use $0", "{s39}"(i32 %sgpr39) #1
    tail call void asm sideeffect "; reg use $0", "{s40}"(i32 %sgpr40) #1
    tail call void asm sideeffect "; reg use $0", "{s41}"(i32 %sgpr41) #1
    tail call void asm sideeffect "; reg use $0", "{s42}"(i32 %sgpr42) #1
    tail call void asm sideeffect "; reg use $0", "{s43}"(i32 %sgpr43) #1
    tail call void asm sideeffect "; reg use $0", "{s44}"(i32 %sgpr44) #1
    tail call void asm sideeffect "; reg use $0", "{s45}"(i32 %sgpr45) #1
    tail call void asm sideeffect "; reg use $0", "{s46}"(i32 %sgpr46) #1
    tail call void asm sideeffect "; reg use $0", "{s47}"(i32 %sgpr47) #1
    tail call void asm sideeffect "; reg use $0", "{s48}"(i32 %sgpr48) #1
    tail call void asm sideeffect "; reg use $0", "{s49}"(i32 %sgpr49) #1
    tail call void asm sideeffect "; reg use $0", "{s50}"(i32 %sgpr50) #1
    tail call void asm sideeffect "; reg use $0", "{s51}"(i32 %sgpr51) #1
    tail call void asm sideeffect "; reg use $0", "{s52}"(i32 %sgpr52) #1
    tail call void asm sideeffect "; reg use $0", "{s53}"(i32 %sgpr53) #1
    tail call void asm sideeffect "; reg use $0", "{s54}"(i32 %sgpr54) #1
    tail call void asm sideeffect "; reg use $0", "{s55}"(i32 %sgpr55) #1
    tail call void asm sideeffect "; reg use $0", "{s56}"(i32 %sgpr56) #1
    tail call void asm sideeffect "; reg use $0", "{s57}"(i32 %sgpr57) #1
    tail call void asm sideeffect "; reg use $0", "{s58}"(i32 %sgpr58) #1
    tail call void asm sideeffect "; reg use $0", "{s59}"(i32 %sgpr59) #1
    tail call void asm sideeffect "; reg use $0", "{s60}"(i32 %sgpr60) #1
    tail call void asm sideeffect "; reg use $0", "{s61}"(i32 %sgpr61) #1
    tail call void asm sideeffect "; reg use $0", "{s62}"(i32 %sgpr62) #1
    tail call void asm sideeffect "; reg use $0", "{s63}"(i32 %sgpr63) #1
    tail call void asm sideeffect "; reg use $0", "{s64}"(i32 %sgpr64) #1
    tail call void asm sideeffect "; reg use $0", "{s65}"(i32 %sgpr65) #1
    tail call void asm sideeffect "; reg use $0", "{s66}"(i32 %sgpr66) #1
    tail call void asm sideeffect "; reg use $0", "{s67}"(i32 %sgpr67) #1
    tail call void asm sideeffect "; reg use $0", "{s68}"(i32 %sgpr68) #1
    tail call void asm sideeffect "; reg use $0", "{s69}"(i32 %sgpr69) #1
    tail call void asm sideeffect "; reg use $0", "{s70}"(i32 %sgpr70) #1
    tail call void asm sideeffect "; reg use $0", "{s71}"(i32 %sgpr71) #1
    tail call void asm sideeffect "; reg use $0", "{s72}"(i32 %sgpr72) #1
    tail call void asm sideeffect "; reg use $0", "{s73}"(i32 %sgpr73) #1
    tail call void asm sideeffect "; reg use $0", "{s74}"(i32 %sgpr74) #1
    tail call void asm sideeffect "; reg use $0", "{s75}"(i32 %sgpr75) #1
    tail call void asm sideeffect "; reg use $0", "{s76}"(i32 %sgpr76) #1
    tail call void asm sideeffect "; reg use $0", "{s77}"(i32 %sgpr77) #1
    tail call void asm sideeffect "; reg use $0", "{s78}"(i32 %sgpr78) #1
    tail call void asm sideeffect "; reg use $0", "{s79}"(i32 %sgpr79) #1
    tail call void asm sideeffect "; reg use $0", "{s80}"(i32 %sgpr80) #1
    tail call void asm sideeffect "; reg use $0", "{s81}"(i32 %sgpr81) #1
    tail call void asm sideeffect "; reg use $0", "{s82}"(i32 %sgpr82) #1
    tail call void asm sideeffect "; reg use $0", "{s83}"(i32 %sgpr83) #1
    tail call void asm sideeffect "; reg use $0", "{s84}"(i32 %sgpr84) #1
    tail call void asm sideeffect "; reg use $0", "{s85}"(i32 %sgpr85) #1
    tail call void asm sideeffect "; reg use $0", "{s86}"(i32 %sgpr86) #1
    tail call void asm sideeffect "; reg use $0", "{s87}"(i32 %sgpr87) #1
    tail call void asm sideeffect "; reg use $0", "{s88}"(i32 %sgpr88) #1
    tail call void asm sideeffect "; reg use $0", "{s89}"(i32 %sgpr89) #1
    tail call void asm sideeffect "; reg use $0", "{s90}"(i32 %sgpr90) #1
    tail call void asm sideeffect "; reg use $0", "{s91}"(i32 %sgpr91) #1
    tail call void asm sideeffect "; reg use $0", "{s92}"(i32 %sgpr92) #1
    tail call void asm sideeffect "; reg use $0", "{s93}"(i32 %sgpr93) #1
    tail call void asm sideeffect "; reg use $0", "{s94}"(i32 %sgpr94) #1
    tail call void asm sideeffect "; reg use $0", "{s95}"(i32 %sgpr95) #1
    tail call void asm sideeffect "; reg use $0", "{s96}"(i32 %sgpr96) #1
    tail call void asm sideeffect "; reg use $0", "{s97}"(i32 %sgpr97) #1
    tail call void asm sideeffect "; reg use $0", "{vcc_lo}"(i32 %vcc_lo) #1
    tail call void asm sideeffect "; reg use $0", "{vcc_hi}"(i32 %vcc_hi) #1
    ret void
  }
; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare align 4 ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #2

attributes #0 = { "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }
attributes #1 = { nounwind }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
!0 = !{}
