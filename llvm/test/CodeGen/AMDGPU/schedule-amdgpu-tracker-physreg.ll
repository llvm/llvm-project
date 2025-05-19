; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tahiti -amdgpu-s-branch-bits=5 -amdgpu-long-branch-factor=0  < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tahiti -amdgpu-s-branch-bits=5 -amdgpu-long-branch-factor=0 -amdgpu-use-amdgpu-trackers=1  < %s | FileCheck --check-prefix=GCN-GCNTRACKERS %s

; CHECK-LABEL: {{^}}spill:
; GCN:    NumSgprs: 104
; GCN-GCNTRACKERS:    NumSgprs: 104
; GCN:    NumVgprs: 1
; GCN-GCNTRACKERS:    NumVgprs: 2
; GCN:    ScratchSize: 0
; GCN-GCNTRACKERS:    ScratchSize: 0
; GCN:    Occupancy: 5
; GCN-GCNTRACKERS:    Occupancy: 5

; FIXME: GCN Trackers do not track pressure from PhysRegs, so scheduling is actually worse

define amdgpu_kernel void @spill(ptr addrspace(1) %arg, i32 %cnd) #0 {
entry:
  %sgpr0 = tail call i32 asm sideeffect "s_mov_b32 s0, 0", "={s0}"() #0
  %sgpr1 = tail call i32 asm sideeffect "s_mov_b32 s1, 0", "={s1}"() #0
  %sgpr2 = tail call i32 asm sideeffect "s_mov_b32 s2, 0", "={s2}"() #0
  %sgpr3 = tail call i32 asm sideeffect "s_mov_b32 s3, 0", "={s3}"() #0
  %sgpr4 = tail call i32 asm sideeffect "s_mov_b32 s4, 0", "={s4}"() #0
  %sgpr5 = tail call i32 asm sideeffect "s_mov_b32 s5, 0", "={s5}"() #0
  %sgpr6 = tail call i32 asm sideeffect "s_mov_b32 s6, 0", "={s6}"() #0
  %sgpr7 = tail call i32 asm sideeffect "s_mov_b32 s7, 0", "={s7}"() #0
  %sgpr8 = tail call i32 asm sideeffect "s_mov_b32 s8, 0", "={s8}"() #0
  %sgpr9 = tail call i32 asm sideeffect "s_mov_b32 s9, 0", "={s9}"() #0
  %sgpr10 = tail call i32 asm sideeffect "s_mov_b32 s10, 0", "={s10}"() #0
  %sgpr11 = tail call i32 asm sideeffect "s_mov_b32 s11, 0", "={s11}"() #0
  %sgpr12 = tail call i32 asm sideeffect "s_mov_b32 s12, 0", "={s12}"() #0
  %sgpr13 = tail call i32 asm sideeffect "s_mov_b32 s13, 0", "={s13}"() #0
  %sgpr14 = tail call i32 asm sideeffect "s_mov_b32 s14, 0", "={s14}"() #0
  %sgpr15 = tail call i32 asm sideeffect "s_mov_b32 s15, 0", "={s15}"() #0
  %sgpr16 = tail call i32 asm sideeffect "s_mov_b32 s16, 0", "={s16}"() #0
  %sgpr17 = tail call i32 asm sideeffect "s_mov_b32 s17, 0", "={s17}"() #0
  %sgpr18 = tail call i32 asm sideeffect "s_mov_b32 s18, 0", "={s18}"() #0
  %sgpr19 = tail call i32 asm sideeffect "s_mov_b32 s19, 0", "={s19}"() #0
  %sgpr20 = tail call i32 asm sideeffect "s_mov_b32 s20, 0", "={s20}"() #0
  %sgpr21 = tail call i32 asm sideeffect "s_mov_b32 s21, 0", "={s21}"() #0
  %sgpr22 = tail call i32 asm sideeffect "s_mov_b32 s22, 0", "={s22}"() #0
  %sgpr23 = tail call i32 asm sideeffect "s_mov_b32 s23, 0", "={s23}"() #0
  %sgpr24 = tail call i32 asm sideeffect "s_mov_b32 s24, 0", "={s24}"() #0
  %sgpr25 = tail call i32 asm sideeffect "s_mov_b32 s25, 0", "={s25}"() #0
  %sgpr26 = tail call i32 asm sideeffect "s_mov_b32 s26, 0", "={s26}"() #0
  %sgpr27 = tail call i32 asm sideeffect "s_mov_b32 s27, 0", "={s27}"() #0
  %sgpr28 = tail call i32 asm sideeffect "s_mov_b32 s28, 0", "={s28}"() #0
  %sgpr29 = tail call i32 asm sideeffect "s_mov_b32 s29, 0", "={s29}"() #0
  %sgpr30 = tail call i32 asm sideeffect "s_mov_b32 s30, 0", "={s30}"() #0
  %sgpr31 = tail call i32 asm sideeffect "s_mov_b32 s31, 0", "={s31}"() #0
  %sgpr32 = tail call i32 asm sideeffect "s_mov_b32 s32, 0", "={s32}"() #0
  %sgpr33 = tail call i32 asm sideeffect "s_mov_b32 s33, 0", "={s33}"() #0
  %sgpr34 = tail call i32 asm sideeffect "s_mov_b32 s34, 0", "={s34}"() #0
  %sgpr35 = tail call i32 asm sideeffect "s_mov_b32 s35, 0", "={s35}"() #0
  %sgpr36 = tail call i32 asm sideeffect "s_mov_b32 s36, 0", "={s36}"() #0
  %sgpr37 = tail call i32 asm sideeffect "s_mov_b32 s37, 0", "={s37}"() #0
  %sgpr38 = tail call i32 asm sideeffect "s_mov_b32 s38, 0", "={s38}"() #0
  %sgpr39 = tail call i32 asm sideeffect "s_mov_b32 s39, 0", "={s39}"() #0
  %sgpr40 = tail call i32 asm sideeffect "s_mov_b32 s40, 0", "={s40}"() #0
  %sgpr41 = tail call i32 asm sideeffect "s_mov_b32 s41, 0", "={s41}"() #0
  %sgpr42 = tail call i32 asm sideeffect "s_mov_b32 s42, 0", "={s42}"() #0
  %sgpr43 = tail call i32 asm sideeffect "s_mov_b32 s43, 0", "={s43}"() #0
  %sgpr44 = tail call i32 asm sideeffect "s_mov_b32 s44, 0", "={s44}"() #0
  %sgpr45 = tail call i32 asm sideeffect "s_mov_b32 s45, 0", "={s45}"() #0
  %sgpr46 = tail call i32 asm sideeffect "s_mov_b32 s46, 0", "={s46}"() #0
  %sgpr47 = tail call i32 asm sideeffect "s_mov_b32 s47, 0", "={s47}"() #0
  %sgpr48 = tail call i32 asm sideeffect "s_mov_b32 s48, 0", "={s48}"() #0
  %sgpr49 = tail call i32 asm sideeffect "s_mov_b32 s49, 0", "={s49}"() #0
  %sgpr50 = tail call i32 asm sideeffect "s_mov_b32 s50, 0", "={s50}"() #0
  %sgpr51 = tail call i32 asm sideeffect "s_mov_b32 s51, 0", "={s51}"() #0
  %sgpr52 = tail call i32 asm sideeffect "s_mov_b32 s52, 0", "={s52}"() #0
  %sgpr53 = tail call i32 asm sideeffect "s_mov_b32 s53, 0", "={s53}"() #0
  %sgpr54 = tail call i32 asm sideeffect "s_mov_b32 s54, 0", "={s54}"() #0
  %sgpr55 = tail call i32 asm sideeffect "s_mov_b32 s55, 0", "={s55}"() #0
  %sgpr56 = tail call i32 asm sideeffect "s_mov_b32 s56, 0", "={s56}"() #0
  %sgpr57 = tail call i32 asm sideeffect "s_mov_b32 s57, 0", "={s57}"() #0
  %sgpr58 = tail call i32 asm sideeffect "s_mov_b32 s58, 0", "={s58}"() #0
  %sgpr59 = tail call i32 asm sideeffect "s_mov_b32 s59, 0", "={s59}"() #0
  %sgpr60 = tail call i32 asm sideeffect "s_mov_b32 s60, 0", "={s60}"() #0
  %sgpr61 = tail call i32 asm sideeffect "s_mov_b32 s61, 0", "={s61}"() #0
  %sgpr62 = tail call i32 asm sideeffect "s_mov_b32 s62, 0", "={s62}"() #0
  %sgpr63 = tail call i32 asm sideeffect "s_mov_b32 s63, 0", "={s63}"() #0
  %sgpr64 = tail call i32 asm sideeffect "s_mov_b32 s64, 0", "={s64}"() #0
  %sgpr65 = tail call i32 asm sideeffect "s_mov_b32 s65, 0", "={s65}"() #0
  %sgpr66 = tail call i32 asm sideeffect "s_mov_b32 s66, 0", "={s66}"() #0
  %sgpr67 = tail call i32 asm sideeffect "s_mov_b32 s67, 0", "={s67}"() #0
  %sgpr68 = tail call i32 asm sideeffect "s_mov_b32 s68, 0", "={s68}"() #0
  %sgpr69 = tail call i32 asm sideeffect "s_mov_b32 s69, 0", "={s69}"() #0
  %sgpr70 = tail call i32 asm sideeffect "s_mov_b32 s70, 0", "={s70}"() #0
  %sgpr71 = tail call i32 asm sideeffect "s_mov_b32 s71, 0", "={s71}"() #0
  %sgpr72 = tail call i32 asm sideeffect "s_mov_b32 s72, 0", "={s72}"() #0
  %sgpr73 = tail call i32 asm sideeffect "s_mov_b32 s73, 0", "={s73}"() #0
  %sgpr74 = tail call i32 asm sideeffect "s_mov_b32 s74, 0", "={s74}"() #0
  %sgpr75 = tail call i32 asm sideeffect "s_mov_b32 s75, 0", "={s75}"() #0
  %sgpr76 = tail call i32 asm sideeffect "s_mov_b32 s76, 0", "={s76}"() #0
  %sgpr77 = tail call i32 asm sideeffect "s_mov_b32 s77, 0", "={s77}"() #0
  %sgpr78 = tail call i32 asm sideeffect "s_mov_b32 s78, 0", "={s78}"() #0
  %sgpr79 = tail call i32 asm sideeffect "s_mov_b32 s79, 0", "={s79}"() #0
  %sgpr80 = tail call i32 asm sideeffect "s_mov_b32 s80, 0", "={s80}"() #0
  %sgpr81 = tail call i32 asm sideeffect "s_mov_b32 s81, 0", "={s81}"() #0
  %sgpr82 = tail call i32 asm sideeffect "s_mov_b32 s82, 0", "={s82}"() #0
  %sgpr83 = tail call i32 asm sideeffect "s_mov_b32 s83, 0", "={s83}"() #0
  %sgpr84 = tail call i32 asm sideeffect "s_mov_b32 s84, 0", "={s84}"() #0
  %sgpr85 = tail call i32 asm sideeffect "s_mov_b32 s85, 0", "={s85}"() #0
  %sgpr86 = tail call i32 asm sideeffect "s_mov_b32 s86, 0", "={s86}"() #0
  %sgpr87 = tail call i32 asm sideeffect "s_mov_b32 s87, 0", "={s87}"() #0
  %sgpr88 = tail call i32 asm sideeffect "s_mov_b32 s88, 0", "={s88}"() #0
  %sgpr89 = tail call i32 asm sideeffect "s_mov_b32 s89, 0", "={s89}"() #0
  %sgpr90 = tail call i32 asm sideeffect "s_mov_b32 s90, 0", "={s90}"() #0
  %sgpr91 = tail call i32 asm sideeffect "s_mov_b32 s91, 0", "={s91}"() #0
  %sgpr92 = tail call i32 asm sideeffect "s_mov_b32 s92, 0", "={s92}"() #0
  %sgpr93 = tail call i32 asm sideeffect "s_mov_b32 s93, 0", "={s93}"() #0
  %sgpr94 = tail call i32 asm sideeffect "s_mov_b32 s94, 0", "={s94}"() #0
  %sgpr95 = tail call i32 asm sideeffect "s_mov_b32 s95, 0", "={s95}"() #0
  %sgpr96 = tail call i32 asm sideeffect "s_mov_b32 s96, 0", "={s96}"() #0
  %sgpr97 = tail call i32 asm sideeffect "s_mov_b32 s97, 0", "={s97}"() #0
  %sgpr98 = tail call i32 asm sideeffect "s_mov_b32 s98, 0", "={s98}"() #0
  %sgpr99 = tail call i32 asm sideeffect "s_mov_b32 s99, 0", "={s99}"() #0
  %sgpr100 = tail call i32 asm sideeffect "s_mov_b32 s100, 0", "={s100}"() #0
  %sgpr101 = tail call i32 asm sideeffect "s_mov_b32 s101, 0", "={s101}"() #0
  %vcc_lo = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={vcc_lo}"() #0
  %vcc_hi = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={vcc_hi}"() #0
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2 ; +8 dword branch

bb2: ; 68 bytes
  ; 64 byte asm
  call void asm sideeffect
   "v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64",""() #0
  br label %bb3

bb3:
  tail call void asm sideeffect "; reg use $0", "{s0}"(i32 %sgpr0) #0
  tail call void asm sideeffect "; reg use $0", "{s1}"(i32 %sgpr1) #0
  tail call void asm sideeffect "; reg use $0", "{s2}"(i32 %sgpr2) #0
  tail call void asm sideeffect "; reg use $0", "{s3}"(i32 %sgpr3) #0
  tail call void asm sideeffect "; reg use $0", "{s4}"(i32 %sgpr4) #0
  tail call void asm sideeffect "; reg use $0", "{s5}"(i32 %sgpr5) #0
  tail call void asm sideeffect "; reg use $0", "{s6}"(i32 %sgpr6) #0
  tail call void asm sideeffect "; reg use $0", "{s7}"(i32 %sgpr7) #0
  tail call void asm sideeffect "; reg use $0", "{s8}"(i32 %sgpr8) #0
  tail call void asm sideeffect "; reg use $0", "{s9}"(i32 %sgpr9) #0
  tail call void asm sideeffect "; reg use $0", "{s10}"(i32 %sgpr10) #0
  tail call void asm sideeffect "; reg use $0", "{s11}"(i32 %sgpr11) #0
  tail call void asm sideeffect "; reg use $0", "{s12}"(i32 %sgpr12) #0
  tail call void asm sideeffect "; reg use $0", "{s13}"(i32 %sgpr13) #0
  tail call void asm sideeffect "; reg use $0", "{s14}"(i32 %sgpr14) #0
  tail call void asm sideeffect "; reg use $0", "{s15}"(i32 %sgpr15) #0
  tail call void asm sideeffect "; reg use $0", "{s16}"(i32 %sgpr16) #0
  tail call void asm sideeffect "; reg use $0", "{s17}"(i32 %sgpr17) #0
  tail call void asm sideeffect "; reg use $0", "{s18}"(i32 %sgpr18) #0
  tail call void asm sideeffect "; reg use $0", "{s19}"(i32 %sgpr19) #0
  tail call void asm sideeffect "; reg use $0", "{s20}"(i32 %sgpr20) #0
  tail call void asm sideeffect "; reg use $0", "{s21}"(i32 %sgpr21) #0
  tail call void asm sideeffect "; reg use $0", "{s22}"(i32 %sgpr22) #0
  tail call void asm sideeffect "; reg use $0", "{s23}"(i32 %sgpr23) #0
  tail call void asm sideeffect "; reg use $0", "{s24}"(i32 %sgpr24) #0
  tail call void asm sideeffect "; reg use $0", "{s25}"(i32 %sgpr25) #0
  tail call void asm sideeffect "; reg use $0", "{s26}"(i32 %sgpr26) #0
  tail call void asm sideeffect "; reg use $0", "{s27}"(i32 %sgpr27) #0
  tail call void asm sideeffect "; reg use $0", "{s28}"(i32 %sgpr28) #0
  tail call void asm sideeffect "; reg use $0", "{s29}"(i32 %sgpr29) #0
  tail call void asm sideeffect "; reg use $0", "{s30}"(i32 %sgpr30) #0
  tail call void asm sideeffect "; reg use $0", "{s31}"(i32 %sgpr31) #0
  tail call void asm sideeffect "; reg use $0", "{s32}"(i32 %sgpr32) #0
  tail call void asm sideeffect "; reg use $0", "{s33}"(i32 %sgpr33) #0
  tail call void asm sideeffect "; reg use $0", "{s34}"(i32 %sgpr34) #0
  tail call void asm sideeffect "; reg use $0", "{s35}"(i32 %sgpr35) #0
  tail call void asm sideeffect "; reg use $0", "{s36}"(i32 %sgpr36) #0
  tail call void asm sideeffect "; reg use $0", "{s37}"(i32 %sgpr37) #0
  tail call void asm sideeffect "; reg use $0", "{s38}"(i32 %sgpr38) #0
  tail call void asm sideeffect "; reg use $0", "{s39}"(i32 %sgpr39) #0
  tail call void asm sideeffect "; reg use $0", "{s40}"(i32 %sgpr40) #0
  tail call void asm sideeffect "; reg use $0", "{s41}"(i32 %sgpr41) #0
  tail call void asm sideeffect "; reg use $0", "{s42}"(i32 %sgpr42) #0
  tail call void asm sideeffect "; reg use $0", "{s43}"(i32 %sgpr43) #0
  tail call void asm sideeffect "; reg use $0", "{s44}"(i32 %sgpr44) #0
  tail call void asm sideeffect "; reg use $0", "{s45}"(i32 %sgpr45) #0
  tail call void asm sideeffect "; reg use $0", "{s46}"(i32 %sgpr46) #0
  tail call void asm sideeffect "; reg use $0", "{s47}"(i32 %sgpr47) #0
  tail call void asm sideeffect "; reg use $0", "{s48}"(i32 %sgpr48) #0
  tail call void asm sideeffect "; reg use $0", "{s49}"(i32 %sgpr49) #0
  tail call void asm sideeffect "; reg use $0", "{s50}"(i32 %sgpr50) #0
  tail call void asm sideeffect "; reg use $0", "{s51}"(i32 %sgpr51) #0
  tail call void asm sideeffect "; reg use $0", "{s52}"(i32 %sgpr52) #0
  tail call void asm sideeffect "; reg use $0", "{s53}"(i32 %sgpr53) #0
  tail call void asm sideeffect "; reg use $0", "{s54}"(i32 %sgpr54) #0
  tail call void asm sideeffect "; reg use $0", "{s55}"(i32 %sgpr55) #0
  tail call void asm sideeffect "; reg use $0", "{s56}"(i32 %sgpr56) #0
  tail call void asm sideeffect "; reg use $0", "{s57}"(i32 %sgpr57) #0
  tail call void asm sideeffect "; reg use $0", "{s58}"(i32 %sgpr58) #0
  tail call void asm sideeffect "; reg use $0", "{s59}"(i32 %sgpr59) #0
  tail call void asm sideeffect "; reg use $0", "{s60}"(i32 %sgpr60) #0
  tail call void asm sideeffect "; reg use $0", "{s61}"(i32 %sgpr61) #0
  tail call void asm sideeffect "; reg use $0", "{s62}"(i32 %sgpr62) #0
  tail call void asm sideeffect "; reg use $0", "{s63}"(i32 %sgpr63) #0
  tail call void asm sideeffect "; reg use $0", "{s64}"(i32 %sgpr64) #0
  tail call void asm sideeffect "; reg use $0", "{s65}"(i32 %sgpr65) #0
  tail call void asm sideeffect "; reg use $0", "{s66}"(i32 %sgpr66) #0
  tail call void asm sideeffect "; reg use $0", "{s67}"(i32 %sgpr67) #0
  tail call void asm sideeffect "; reg use $0", "{s68}"(i32 %sgpr68) #0
  tail call void asm sideeffect "; reg use $0", "{s69}"(i32 %sgpr69) #0
  tail call void asm sideeffect "; reg use $0", "{s70}"(i32 %sgpr70) #0
  tail call void asm sideeffect "; reg use $0", "{s71}"(i32 %sgpr71) #0
  tail call void asm sideeffect "; reg use $0", "{s72}"(i32 %sgpr72) #0
  tail call void asm sideeffect "; reg use $0", "{s73}"(i32 %sgpr73) #0
  tail call void asm sideeffect "; reg use $0", "{s74}"(i32 %sgpr74) #0
  tail call void asm sideeffect "; reg use $0", "{s75}"(i32 %sgpr75) #0
  tail call void asm sideeffect "; reg use $0", "{s76}"(i32 %sgpr76) #0
  tail call void asm sideeffect "; reg use $0", "{s77}"(i32 %sgpr77) #0
  tail call void asm sideeffect "; reg use $0", "{s78}"(i32 %sgpr78) #0
  tail call void asm sideeffect "; reg use $0", "{s79}"(i32 %sgpr79) #0
  tail call void asm sideeffect "; reg use $0", "{s80}"(i32 %sgpr80) #0
  tail call void asm sideeffect "; reg use $0", "{s81}"(i32 %sgpr81) #0
  tail call void asm sideeffect "; reg use $0", "{s82}"(i32 %sgpr82) #0
  tail call void asm sideeffect "; reg use $0", "{s83}"(i32 %sgpr83) #0
  tail call void asm sideeffect "; reg use $0", "{s84}"(i32 %sgpr84) #0
  tail call void asm sideeffect "; reg use $0", "{s85}"(i32 %sgpr85) #0
  tail call void asm sideeffect "; reg use $0", "{s86}"(i32 %sgpr86) #0
  tail call void asm sideeffect "; reg use $0", "{s87}"(i32 %sgpr87) #0
  tail call void asm sideeffect "; reg use $0", "{s88}"(i32 %sgpr88) #0
  tail call void asm sideeffect "; reg use $0", "{s89}"(i32 %sgpr89) #0
  tail call void asm sideeffect "; reg use $0", "{s90}"(i32 %sgpr90) #0
  tail call void asm sideeffect "; reg use $0", "{s91}"(i32 %sgpr91) #0
  tail call void asm sideeffect "; reg use $0", "{s92}"(i32 %sgpr92) #0
  tail call void asm sideeffect "; reg use $0", "{s93}"(i32 %sgpr93) #0
  tail call void asm sideeffect "; reg use $0", "{s94}"(i32 %sgpr94) #0
  tail call void asm sideeffect "; reg use $0", "{s95}"(i32 %sgpr95) #0
  tail call void asm sideeffect "; reg use $0", "{s96}"(i32 %sgpr96) #0
  tail call void asm sideeffect "; reg use $0", "{s97}"(i32 %sgpr97) #0
  tail call void asm sideeffect "; reg use $0", "{s98}"(i32 %sgpr98) #0
  tail call void asm sideeffect "; reg use $0", "{s99}"(i32 %sgpr99) #0
  tail call void asm sideeffect "; reg use $0", "{s100}"(i32 %sgpr100) #0
  tail call void asm sideeffect "; reg use $0", "{s101}"(i32 %sgpr101) #0
  tail call void asm sideeffect "; reg use $0", "{vcc_lo}"(i32 %vcc_lo) #0
  tail call void asm sideeffect "; reg use $0", "{vcc_hi}"(i32 %vcc_hi) #0
  ret void
}

; CHECK-LABEL: {{^}}spill_func:
; GCN:    NumSgprs: 104
; GCN-GCNTRACKERS:    NumSgprs: 104
; GCN:    NumVgprs: 2
; GCN-GCNTRACKERS:    NumVgprs: 3
; GCN:    ScratchSize: 8
; GCN-GCNTRACKERS:    ScratchSize: 12

define void @spill_func(ptr addrspace(1) %arg) #0 {
entry:
  %cnd = tail call i32 @llvm.amdgcn.workgroup.id.x() #0
  %sgpr0 = tail call i32 asm sideeffect "s_mov_b32 s0, 0", "={s0}"() #0
  %sgpr1 = tail call i32 asm sideeffect "s_mov_b32 s1, 0", "={s1}"() #0
  %sgpr2 = tail call i32 asm sideeffect "s_mov_b32 s2, 0", "={s2}"() #0
  %sgpr3 = tail call i32 asm sideeffect "s_mov_b32 s3, 0", "={s3}"() #0
  %sgpr4 = tail call i32 asm sideeffect "s_mov_b32 s4, 0", "={s4}"() #0
  %sgpr5 = tail call i32 asm sideeffect "s_mov_b32 s5, 0", "={s5}"() #0
  %sgpr6 = tail call i32 asm sideeffect "s_mov_b32 s6, 0", "={s6}"() #0
  %sgpr7 = tail call i32 asm sideeffect "s_mov_b32 s7, 0", "={s7}"() #0
  %sgpr8 = tail call i32 asm sideeffect "s_mov_b32 s8, 0", "={s8}"() #0
  %sgpr9 = tail call i32 asm sideeffect "s_mov_b32 s9, 0", "={s9}"() #0
  %sgpr10 = tail call i32 asm sideeffect "s_mov_b32 s10, 0", "={s10}"() #0
  %sgpr11 = tail call i32 asm sideeffect "s_mov_b32 s11, 0", "={s11}"() #0
  %sgpr12 = tail call i32 asm sideeffect "s_mov_b32 s12, 0", "={s12}"() #0
  %sgpr13 = tail call i32 asm sideeffect "s_mov_b32 s13, 0", "={s13}"() #0
  %sgpr14 = tail call i32 asm sideeffect "s_mov_b32 s14, 0", "={s14}"() #0
  %sgpr15 = tail call i32 asm sideeffect "s_mov_b32 s15, 0", "={s15}"() #0
  %sgpr16 = tail call i32 asm sideeffect "s_mov_b32 s16, 0", "={s16}"() #0
  %sgpr17 = tail call i32 asm sideeffect "s_mov_b32 s17, 0", "={s17}"() #0
  %sgpr18 = tail call i32 asm sideeffect "s_mov_b32 s18, 0", "={s18}"() #0
  %sgpr19 = tail call i32 asm sideeffect "s_mov_b32 s19, 0", "={s19}"() #0
  %sgpr20 = tail call i32 asm sideeffect "s_mov_b32 s20, 0", "={s20}"() #0
  %sgpr21 = tail call i32 asm sideeffect "s_mov_b32 s21, 0", "={s21}"() #0
  %sgpr22 = tail call i32 asm sideeffect "s_mov_b32 s22, 0", "={s22}"() #0
  %sgpr23 = tail call i32 asm sideeffect "s_mov_b32 s23, 0", "={s23}"() #0
  %sgpr24 = tail call i32 asm sideeffect "s_mov_b32 s24, 0", "={s24}"() #0
  %sgpr25 = tail call i32 asm sideeffect "s_mov_b32 s25, 0", "={s25}"() #0
  %sgpr26 = tail call i32 asm sideeffect "s_mov_b32 s26, 0", "={s26}"() #0
  %sgpr27 = tail call i32 asm sideeffect "s_mov_b32 s27, 0", "={s27}"() #0
  %sgpr28 = tail call i32 asm sideeffect "s_mov_b32 s28, 0", "={s28}"() #0
  %sgpr29 = tail call i32 asm sideeffect "s_mov_b32 s29, 0", "={s29}"() #0
  %sgpr30 = tail call i32 asm sideeffect "s_mov_b32 s30, 0", "={s30}"() #0
  %sgpr31 = tail call i32 asm sideeffect "s_mov_b32 s31, 0", "={s31}"() #0
  %sgpr32 = tail call i32 asm sideeffect "s_mov_b32 s32, 0", "={s32}"() #0
  %sgpr33 = tail call i32 asm sideeffect "s_mov_b32 s33, 0", "={s33}"() #0
  %sgpr34 = tail call i32 asm sideeffect "s_mov_b32 s34, 0", "={s34}"() #0
  %sgpr35 = tail call i32 asm sideeffect "s_mov_b32 s35, 0", "={s35}"() #0
  %sgpr36 = tail call i32 asm sideeffect "s_mov_b32 s36, 0", "={s36}"() #0
  %sgpr37 = tail call i32 asm sideeffect "s_mov_b32 s37, 0", "={s37}"() #0
  %sgpr38 = tail call i32 asm sideeffect "s_mov_b32 s38, 0", "={s38}"() #0
  %sgpr39 = tail call i32 asm sideeffect "s_mov_b32 s39, 0", "={s39}"() #0
  %sgpr40 = tail call i32 asm sideeffect "s_mov_b32 s40, 0", "={s40}"() #0
  %sgpr41 = tail call i32 asm sideeffect "s_mov_b32 s41, 0", "={s41}"() #0
  %sgpr42 = tail call i32 asm sideeffect "s_mov_b32 s42, 0", "={s42}"() #0
  %sgpr43 = tail call i32 asm sideeffect "s_mov_b32 s43, 0", "={s43}"() #0
  %sgpr44 = tail call i32 asm sideeffect "s_mov_b32 s44, 0", "={s44}"() #0
  %sgpr45 = tail call i32 asm sideeffect "s_mov_b32 s45, 0", "={s45}"() #0
  %sgpr46 = tail call i32 asm sideeffect "s_mov_b32 s46, 0", "={s46}"() #0
  %sgpr47 = tail call i32 asm sideeffect "s_mov_b32 s47, 0", "={s47}"() #0
  %sgpr48 = tail call i32 asm sideeffect "s_mov_b32 s48, 0", "={s48}"() #0
  %sgpr49 = tail call i32 asm sideeffect "s_mov_b32 s49, 0", "={s49}"() #0
  %sgpr50 = tail call i32 asm sideeffect "s_mov_b32 s50, 0", "={s50}"() #0
  %sgpr51 = tail call i32 asm sideeffect "s_mov_b32 s51, 0", "={s51}"() #0
  %sgpr52 = tail call i32 asm sideeffect "s_mov_b32 s52, 0", "={s52}"() #0
  %sgpr53 = tail call i32 asm sideeffect "s_mov_b32 s53, 0", "={s53}"() #0
  %sgpr54 = tail call i32 asm sideeffect "s_mov_b32 s54, 0", "={s54}"() #0
  %sgpr55 = tail call i32 asm sideeffect "s_mov_b32 s55, 0", "={s55}"() #0
  %sgpr56 = tail call i32 asm sideeffect "s_mov_b32 s56, 0", "={s56}"() #0
  %sgpr57 = tail call i32 asm sideeffect "s_mov_b32 s57, 0", "={s57}"() #0
  %sgpr58 = tail call i32 asm sideeffect "s_mov_b32 s58, 0", "={s58}"() #0
  %sgpr59 = tail call i32 asm sideeffect "s_mov_b32 s59, 0", "={s59}"() #0
  %sgpr60 = tail call i32 asm sideeffect "s_mov_b32 s60, 0", "={s60}"() #0
  %sgpr61 = tail call i32 asm sideeffect "s_mov_b32 s61, 0", "={s61}"() #0
  %sgpr62 = tail call i32 asm sideeffect "s_mov_b32 s62, 0", "={s62}"() #0
  %sgpr63 = tail call i32 asm sideeffect "s_mov_b32 s63, 0", "={s63}"() #0
  %sgpr64 = tail call i32 asm sideeffect "s_mov_b32 s64, 0", "={s64}"() #0
  %sgpr65 = tail call i32 asm sideeffect "s_mov_b32 s65, 0", "={s65}"() #0
  %sgpr66 = tail call i32 asm sideeffect "s_mov_b32 s66, 0", "={s66}"() #0
  %sgpr67 = tail call i32 asm sideeffect "s_mov_b32 s67, 0", "={s67}"() #0
  %sgpr68 = tail call i32 asm sideeffect "s_mov_b32 s68, 0", "={s68}"() #0
  %sgpr69 = tail call i32 asm sideeffect "s_mov_b32 s69, 0", "={s69}"() #0
  %sgpr70 = tail call i32 asm sideeffect "s_mov_b32 s70, 0", "={s70}"() #0
  %sgpr71 = tail call i32 asm sideeffect "s_mov_b32 s71, 0", "={s71}"() #0
  %sgpr72 = tail call i32 asm sideeffect "s_mov_b32 s72, 0", "={s72}"() #0
  %sgpr73 = tail call i32 asm sideeffect "s_mov_b32 s73, 0", "={s73}"() #0
  %sgpr74 = tail call i32 asm sideeffect "s_mov_b32 s74, 0", "={s74}"() #0
  %sgpr75 = tail call i32 asm sideeffect "s_mov_b32 s75, 0", "={s75}"() #0
  %sgpr76 = tail call i32 asm sideeffect "s_mov_b32 s76, 0", "={s76}"() #0
  %sgpr77 = tail call i32 asm sideeffect "s_mov_b32 s77, 0", "={s77}"() #0
  %sgpr78 = tail call i32 asm sideeffect "s_mov_b32 s78, 0", "={s78}"() #0
  %sgpr79 = tail call i32 asm sideeffect "s_mov_b32 s79, 0", "={s79}"() #0
  %sgpr80 = tail call i32 asm sideeffect "s_mov_b32 s80, 0", "={s80}"() #0
  %sgpr81 = tail call i32 asm sideeffect "s_mov_b32 s81, 0", "={s81}"() #0
  %sgpr82 = tail call i32 asm sideeffect "s_mov_b32 s82, 0", "={s82}"() #0
  %sgpr83 = tail call i32 asm sideeffect "s_mov_b32 s83, 0", "={s83}"() #0
  %sgpr84 = tail call i32 asm sideeffect "s_mov_b32 s84, 0", "={s84}"() #0
  %sgpr85 = tail call i32 asm sideeffect "s_mov_b32 s85, 0", "={s85}"() #0
  %sgpr86 = tail call i32 asm sideeffect "s_mov_b32 s86, 0", "={s86}"() #0
  %sgpr87 = tail call i32 asm sideeffect "s_mov_b32 s87, 0", "={s87}"() #0
  %sgpr88 = tail call i32 asm sideeffect "s_mov_b32 s88, 0", "={s88}"() #0
  %sgpr89 = tail call i32 asm sideeffect "s_mov_b32 s89, 0", "={s89}"() #0
  %sgpr90 = tail call i32 asm sideeffect "s_mov_b32 s90, 0", "={s90}"() #0
  %sgpr91 = tail call i32 asm sideeffect "s_mov_b32 s91, 0", "={s91}"() #0
  %sgpr92 = tail call i32 asm sideeffect "s_mov_b32 s92, 0", "={s92}"() #0
  %sgpr93 = tail call i32 asm sideeffect "s_mov_b32 s93, 0", "={s93}"() #0
  %sgpr94 = tail call i32 asm sideeffect "s_mov_b32 s94, 0", "={s94}"() #0
  %sgpr95 = tail call i32 asm sideeffect "s_mov_b32 s95, 0", "={s95}"() #0
  %sgpr96 = tail call i32 asm sideeffect "s_mov_b32 s96, 0", "={s96}"() #0
  %sgpr97 = tail call i32 asm sideeffect "s_mov_b32 s97, 0", "={s97}"() #0
  %sgpr98 = tail call i32 asm sideeffect "s_mov_b32 s98, 0", "={s98}"() #0
  %sgpr99 = tail call i32 asm sideeffect "s_mov_b32 s99, 0", "={s99}"() #0
  %sgpr100 = tail call i32 asm sideeffect "s_mov_b32 s100, 0", "={s100}"() #0
  %sgpr101 = tail call i32 asm sideeffect "s_mov_b32 s101, 0", "={s101}"() #0
  %vcc_lo = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={vcc_lo}"() #0
  %vcc_hi = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={vcc_hi}"() #0
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2 ; +8 dword branch

bb2: ; 68 bytes
  ; 64 byte asm
  call void asm sideeffect
   "v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64",""() #0
  br label %bb3

bb3:
  tail call void asm sideeffect "; reg use $0", "{s0}"(i32 %sgpr0) #0
  tail call void asm sideeffect "; reg use $0", "{s1}"(i32 %sgpr1) #0
  tail call void asm sideeffect "; reg use $0", "{s2}"(i32 %sgpr2) #0
  tail call void asm sideeffect "; reg use $0", "{s3}"(i32 %sgpr3) #0
  tail call void asm sideeffect "; reg use $0", "{s4}"(i32 %sgpr4) #0
  tail call void asm sideeffect "; reg use $0", "{s5}"(i32 %sgpr5) #0
  tail call void asm sideeffect "; reg use $0", "{s6}"(i32 %sgpr6) #0
  tail call void asm sideeffect "; reg use $0", "{s7}"(i32 %sgpr7) #0
  tail call void asm sideeffect "; reg use $0", "{s8}"(i32 %sgpr8) #0
  tail call void asm sideeffect "; reg use $0", "{s9}"(i32 %sgpr9) #0
  tail call void asm sideeffect "; reg use $0", "{s10}"(i32 %sgpr10) #0
  tail call void asm sideeffect "; reg use $0", "{s11}"(i32 %sgpr11) #0
  tail call void asm sideeffect "; reg use $0", "{s12}"(i32 %sgpr12) #0
  tail call void asm sideeffect "; reg use $0", "{s13}"(i32 %sgpr13) #0
  tail call void asm sideeffect "; reg use $0", "{s14}"(i32 %sgpr14) #0
  tail call void asm sideeffect "; reg use $0", "{s15}"(i32 %sgpr15) #0
  tail call void asm sideeffect "; reg use $0", "{s16}"(i32 %sgpr16) #0
  tail call void asm sideeffect "; reg use $0", "{s17}"(i32 %sgpr17) #0
  tail call void asm sideeffect "; reg use $0", "{s18}"(i32 %sgpr18) #0
  tail call void asm sideeffect "; reg use $0", "{s19}"(i32 %sgpr19) #0
  tail call void asm sideeffect "; reg use $0", "{s20}"(i32 %sgpr20) #0
  tail call void asm sideeffect "; reg use $0", "{s21}"(i32 %sgpr21) #0
  tail call void asm sideeffect "; reg use $0", "{s22}"(i32 %sgpr22) #0
  tail call void asm sideeffect "; reg use $0", "{s23}"(i32 %sgpr23) #0
  tail call void asm sideeffect "; reg use $0", "{s24}"(i32 %sgpr24) #0
  tail call void asm sideeffect "; reg use $0", "{s25}"(i32 %sgpr25) #0
  tail call void asm sideeffect "; reg use $0", "{s26}"(i32 %sgpr26) #0
  tail call void asm sideeffect "; reg use $0", "{s27}"(i32 %sgpr27) #0
  tail call void asm sideeffect "; reg use $0", "{s28}"(i32 %sgpr28) #0
  tail call void asm sideeffect "; reg use $0", "{s29}"(i32 %sgpr29) #0
  tail call void asm sideeffect "; reg use $0", "{s30}"(i32 %sgpr30) #0
  tail call void asm sideeffect "; reg use $0", "{s31}"(i32 %sgpr31) #0
  tail call void asm sideeffect "; reg use $0", "{s32}"(i32 %sgpr32) #0
  tail call void asm sideeffect "; reg use $0", "{s33}"(i32 %sgpr33) #0
  tail call void asm sideeffect "; reg use $0", "{s34}"(i32 %sgpr34) #0
  tail call void asm sideeffect "; reg use $0", "{s35}"(i32 %sgpr35) #0
  tail call void asm sideeffect "; reg use $0", "{s36}"(i32 %sgpr36) #0
  tail call void asm sideeffect "; reg use $0", "{s37}"(i32 %sgpr37) #0
  tail call void asm sideeffect "; reg use $0", "{s38}"(i32 %sgpr38) #0
  tail call void asm sideeffect "; reg use $0", "{s39}"(i32 %sgpr39) #0
  tail call void asm sideeffect "; reg use $0", "{s40}"(i32 %sgpr40) #0
  tail call void asm sideeffect "; reg use $0", "{s41}"(i32 %sgpr41) #0
  tail call void asm sideeffect "; reg use $0", "{s42}"(i32 %sgpr42) #0
  tail call void asm sideeffect "; reg use $0", "{s43}"(i32 %sgpr43) #0
  tail call void asm sideeffect "; reg use $0", "{s44}"(i32 %sgpr44) #0
  tail call void asm sideeffect "; reg use $0", "{s45}"(i32 %sgpr45) #0
  tail call void asm sideeffect "; reg use $0", "{s46}"(i32 %sgpr46) #0
  tail call void asm sideeffect "; reg use $0", "{s47}"(i32 %sgpr47) #0
  tail call void asm sideeffect "; reg use $0", "{s48}"(i32 %sgpr48) #0
  tail call void asm sideeffect "; reg use $0", "{s49}"(i32 %sgpr49) #0
  tail call void asm sideeffect "; reg use $0", "{s50}"(i32 %sgpr50) #0
  tail call void asm sideeffect "; reg use $0", "{s51}"(i32 %sgpr51) #0
  tail call void asm sideeffect "; reg use $0", "{s52}"(i32 %sgpr52) #0
  tail call void asm sideeffect "; reg use $0", "{s53}"(i32 %sgpr53) #0
  tail call void asm sideeffect "; reg use $0", "{s54}"(i32 %sgpr54) #0
  tail call void asm sideeffect "; reg use $0", "{s55}"(i32 %sgpr55) #0
  tail call void asm sideeffect "; reg use $0", "{s56}"(i32 %sgpr56) #0
  tail call void asm sideeffect "; reg use $0", "{s57}"(i32 %sgpr57) #0
  tail call void asm sideeffect "; reg use $0", "{s58}"(i32 %sgpr58) #0
  tail call void asm sideeffect "; reg use $0", "{s59}"(i32 %sgpr59) #0
  tail call void asm sideeffect "; reg use $0", "{s60}"(i32 %sgpr60) #0
  tail call void asm sideeffect "; reg use $0", "{s61}"(i32 %sgpr61) #0
  tail call void asm sideeffect "; reg use $0", "{s62}"(i32 %sgpr62) #0
  tail call void asm sideeffect "; reg use $0", "{s63}"(i32 %sgpr63) #0
  tail call void asm sideeffect "; reg use $0", "{s64}"(i32 %sgpr64) #0
  tail call void asm sideeffect "; reg use $0", "{s65}"(i32 %sgpr65) #0
  tail call void asm sideeffect "; reg use $0", "{s66}"(i32 %sgpr66) #0
  tail call void asm sideeffect "; reg use $0", "{s67}"(i32 %sgpr67) #0
  tail call void asm sideeffect "; reg use $0", "{s68}"(i32 %sgpr68) #0
  tail call void asm sideeffect "; reg use $0", "{s69}"(i32 %sgpr69) #0
  tail call void asm sideeffect "; reg use $0", "{s70}"(i32 %sgpr70) #0
  tail call void asm sideeffect "; reg use $0", "{s71}"(i32 %sgpr71) #0
  tail call void asm sideeffect "; reg use $0", "{s72}"(i32 %sgpr72) #0
  tail call void asm sideeffect "; reg use $0", "{s73}"(i32 %sgpr73) #0
  tail call void asm sideeffect "; reg use $0", "{s74}"(i32 %sgpr74) #0
  tail call void asm sideeffect "; reg use $0", "{s75}"(i32 %sgpr75) #0
  tail call void asm sideeffect "; reg use $0", "{s76}"(i32 %sgpr76) #0
  tail call void asm sideeffect "; reg use $0", "{s77}"(i32 %sgpr77) #0
  tail call void asm sideeffect "; reg use $0", "{s78}"(i32 %sgpr78) #0
  tail call void asm sideeffect "; reg use $0", "{s79}"(i32 %sgpr79) #0
  tail call void asm sideeffect "; reg use $0", "{s80}"(i32 %sgpr80) #0
  tail call void asm sideeffect "; reg use $0", "{s81}"(i32 %sgpr81) #0
  tail call void asm sideeffect "; reg use $0", "{s82}"(i32 %sgpr82) #0
  tail call void asm sideeffect "; reg use $0", "{s83}"(i32 %sgpr83) #0
  tail call void asm sideeffect "; reg use $0", "{s84}"(i32 %sgpr84) #0
  tail call void asm sideeffect "; reg use $0", "{s85}"(i32 %sgpr85) #0
  tail call void asm sideeffect "; reg use $0", "{s86}"(i32 %sgpr86) #0
  tail call void asm sideeffect "; reg use $0", "{s87}"(i32 %sgpr87) #0
  tail call void asm sideeffect "; reg use $0", "{s88}"(i32 %sgpr88) #0
  tail call void asm sideeffect "; reg use $0", "{s89}"(i32 %sgpr89) #0
  tail call void asm sideeffect "; reg use $0", "{s90}"(i32 %sgpr90) #0
  tail call void asm sideeffect "; reg use $0", "{s91}"(i32 %sgpr91) #0
  tail call void asm sideeffect "; reg use $0", "{s92}"(i32 %sgpr92) #0
  tail call void asm sideeffect "; reg use $0", "{s93}"(i32 %sgpr93) #0
  tail call void asm sideeffect "; reg use $0", "{s94}"(i32 %sgpr94) #0
  tail call void asm sideeffect "; reg use $0", "{s95}"(i32 %sgpr95) #0
  tail call void asm sideeffect "; reg use $0", "{s96}"(i32 %sgpr96) #0
  tail call void asm sideeffect "; reg use $0", "{s97}"(i32 %sgpr97) #0
  tail call void asm sideeffect "; reg use $0", "{s98}"(i32 %sgpr98) #0
  tail call void asm sideeffect "; reg use $0", "{s99}"(i32 %sgpr99) #0
  tail call void asm sideeffect "; reg use $0", "{s100}"(i32 %sgpr100) #0
  tail call void asm sideeffect "; reg use $0", "{s101}"(i32 %sgpr101) #0
  tail call void asm sideeffect "; reg use $0", "{vcc_lo}"(i32 %vcc_lo) #0
  tail call void asm sideeffect "; reg use $0", "{vcc_hi}"(i32 %vcc_hi) #0
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x() #0

attributes #0 = { nounwind }
