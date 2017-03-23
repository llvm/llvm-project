; RUN: not llc -march=amdgcn -verify-machineinstrs  -amdgpu-s-branch-bits=4 < %s 2>&1 | FileCheck -check-prefix=FAIL %s

; FIXME: This should be able to compile, but requires inserting an
; extra block to restore the scavenged register.

; FAIL: LLVM ERROR: Error while trying to spill VCC from class SReg_64: Cannot scavenge register without an emergency spill slot!

define amdgpu_kernel void @spill(i32 addrspace(1)* %arg, i32 %cnd) #0 {
entry:
  %sgpr0 = tail call i32 asm sideeffect "s_mov_b32 s0, 0", "={SGPR0}"() #0
  %sgpr1 = tail call i32 asm sideeffect "s_mov_b32 s1, 0", "={SGPR1}"() #0
  %sgpr2 = tail call i32 asm sideeffect "s_mov_b32 s2, 0", "={SGPR2}"() #0
  %sgpr3 = tail call i32 asm sideeffect "s_mov_b32 s3, 0", "={SGPR3}"() #0
  %sgpr4 = tail call i32 asm sideeffect "s_mov_b32 s4, 0", "={SGPR4}"() #0
  %sgpr5 = tail call i32 asm sideeffect "s_mov_b32 s5, 0", "={SGPR5}"() #0
  %sgpr6 = tail call i32 asm sideeffect "s_mov_b32 s6, 0", "={SGPR6}"() #0
  %sgpr7 = tail call i32 asm sideeffect "s_mov_b32 s7, 0", "={SGPR7}"() #0
  %sgpr8 = tail call i32 asm sideeffect "s_mov_b32 s8, 0", "={SGPR8}"() #0
  %sgpr9 = tail call i32 asm sideeffect "s_mov_b32 s9, 0", "={SGPR9}"() #0
  %sgpr10 = tail call i32 asm sideeffect "s_mov_b32 s10, 0", "={SGPR10}"() #0
  %sgpr11 = tail call i32 asm sideeffect "s_mov_b32 s11, 0", "={SGPR11}"() #0
  %sgpr12 = tail call i32 asm sideeffect "s_mov_b32 s12, 0", "={SGPR12}"() #0
  %sgpr13 = tail call i32 asm sideeffect "s_mov_b32 s13, 0", "={SGPR13}"() #0
  %sgpr14 = tail call i32 asm sideeffect "s_mov_b32 s14, 0", "={SGPR14}"() #0
  %sgpr15 = tail call i32 asm sideeffect "s_mov_b32 s15, 0", "={SGPR15}"() #0
  %sgpr16 = tail call i32 asm sideeffect "s_mov_b32 s16, 0", "={SGPR16}"() #0
  %sgpr17 = tail call i32 asm sideeffect "s_mov_b32 s17, 0", "={SGPR17}"() #0
  %sgpr18 = tail call i32 asm sideeffect "s_mov_b32 s18, 0", "={SGPR18}"() #0
  %sgpr19 = tail call i32 asm sideeffect "s_mov_b32 s19, 0", "={SGPR19}"() #0
  %sgpr20 = tail call i32 asm sideeffect "s_mov_b32 s20, 0", "={SGPR20}"() #0
  %sgpr21 = tail call i32 asm sideeffect "s_mov_b32 s21, 0", "={SGPR21}"() #0
  %sgpr22 = tail call i32 asm sideeffect "s_mov_b32 s22, 0", "={SGPR22}"() #0
  %sgpr23 = tail call i32 asm sideeffect "s_mov_b32 s23, 0", "={SGPR23}"() #0
  %sgpr24 = tail call i32 asm sideeffect "s_mov_b32 s24, 0", "={SGPR24}"() #0
  %sgpr25 = tail call i32 asm sideeffect "s_mov_b32 s25, 0", "={SGPR25}"() #0
  %sgpr26 = tail call i32 asm sideeffect "s_mov_b32 s26, 0", "={SGPR26}"() #0
  %sgpr27 = tail call i32 asm sideeffect "s_mov_b32 s27, 0", "={SGPR27}"() #0
  %sgpr28 = tail call i32 asm sideeffect "s_mov_b32 s28, 0", "={SGPR28}"() #0
  %sgpr29 = tail call i32 asm sideeffect "s_mov_b32 s29, 0", "={SGPR29}"() #0
  %sgpr30 = tail call i32 asm sideeffect "s_mov_b32 s30, 0", "={SGPR30}"() #0
  %sgpr31 = tail call i32 asm sideeffect "s_mov_b32 s31, 0", "={SGPR31}"() #0
  %sgpr32 = tail call i32 asm sideeffect "s_mov_b32 s32, 0", "={SGPR32}"() #0
  %sgpr33 = tail call i32 asm sideeffect "s_mov_b32 s33, 0", "={SGPR33}"() #0
  %sgpr34 = tail call i32 asm sideeffect "s_mov_b32 s34, 0", "={SGPR34}"() #0
  %sgpr35 = tail call i32 asm sideeffect "s_mov_b32 s35, 0", "={SGPR35}"() #0
  %sgpr36 = tail call i32 asm sideeffect "s_mov_b32 s36, 0", "={SGPR36}"() #0
  %sgpr37 = tail call i32 asm sideeffect "s_mov_b32 s37, 0", "={SGPR37}"() #0
  %sgpr38 = tail call i32 asm sideeffect "s_mov_b32 s38, 0", "={SGPR38}"() #0
  %sgpr39 = tail call i32 asm sideeffect "s_mov_b32 s39, 0", "={SGPR39}"() #0
  %sgpr40 = tail call i32 asm sideeffect "s_mov_b32 s40, 0", "={SGPR40}"() #0
  %sgpr41 = tail call i32 asm sideeffect "s_mov_b32 s41, 0", "={SGPR41}"() #0
  %sgpr42 = tail call i32 asm sideeffect "s_mov_b32 s42, 0", "={SGPR42}"() #0
  %sgpr43 = tail call i32 asm sideeffect "s_mov_b32 s43, 0", "={SGPR43}"() #0
  %sgpr44 = tail call i32 asm sideeffect "s_mov_b32 s44, 0", "={SGPR44}"() #0
  %sgpr45 = tail call i32 asm sideeffect "s_mov_b32 s45, 0", "={SGPR45}"() #0
  %sgpr46 = tail call i32 asm sideeffect "s_mov_b32 s46, 0", "={SGPR46}"() #0
  %sgpr47 = tail call i32 asm sideeffect "s_mov_b32 s47, 0", "={SGPR47}"() #0
  %sgpr48 = tail call i32 asm sideeffect "s_mov_b32 s48, 0", "={SGPR48}"() #0
  %sgpr49 = tail call i32 asm sideeffect "s_mov_b32 s49, 0", "={SGPR49}"() #0
  %sgpr50 = tail call i32 asm sideeffect "s_mov_b32 s50, 0", "={SGPR50}"() #0
  %sgpr51 = tail call i32 asm sideeffect "s_mov_b32 s51, 0", "={SGPR51}"() #0
  %sgpr52 = tail call i32 asm sideeffect "s_mov_b32 s52, 0", "={SGPR52}"() #0
  %sgpr53 = tail call i32 asm sideeffect "s_mov_b32 s53, 0", "={SGPR53}"() #0
  %sgpr54 = tail call i32 asm sideeffect "s_mov_b32 s54, 0", "={SGPR54}"() #0
  %sgpr55 = tail call i32 asm sideeffect "s_mov_b32 s55, 0", "={SGPR55}"() #0
  %sgpr56 = tail call i32 asm sideeffect "s_mov_b32 s56, 0", "={SGPR56}"() #0
  %sgpr57 = tail call i32 asm sideeffect "s_mov_b32 s57, 0", "={SGPR57}"() #0
  %sgpr58 = tail call i32 asm sideeffect "s_mov_b32 s58, 0", "={SGPR58}"() #0
  %sgpr59 = tail call i32 asm sideeffect "s_mov_b32 s59, 0", "={SGPR59}"() #0
  %sgpr60 = tail call i32 asm sideeffect "s_mov_b32 s60, 0", "={SGPR60}"() #0
  %sgpr61 = tail call i32 asm sideeffect "s_mov_b32 s61, 0", "={SGPR61}"() #0
  %sgpr62 = tail call i32 asm sideeffect "s_mov_b32 s62, 0", "={SGPR62}"() #0
  %sgpr63 = tail call i32 asm sideeffect "s_mov_b32 s63, 0", "={SGPR63}"() #0
  %sgpr64 = tail call i32 asm sideeffect "s_mov_b32 s64, 0", "={SGPR64}"() #0
  %sgpr65 = tail call i32 asm sideeffect "s_mov_b32 s65, 0", "={SGPR65}"() #0
  %sgpr66 = tail call i32 asm sideeffect "s_mov_b32 s66, 0", "={SGPR66}"() #0
  %sgpr67 = tail call i32 asm sideeffect "s_mov_b32 s67, 0", "={SGPR67}"() #0
  %sgpr68 = tail call i32 asm sideeffect "s_mov_b32 s68, 0", "={SGPR68}"() #0
  %sgpr69 = tail call i32 asm sideeffect "s_mov_b32 s69, 0", "={SGPR69}"() #0
  %sgpr70 = tail call i32 asm sideeffect "s_mov_b32 s70, 0", "={SGPR70}"() #0
  %sgpr71 = tail call i32 asm sideeffect "s_mov_b32 s71, 0", "={SGPR71}"() #0
  %sgpr72 = tail call i32 asm sideeffect "s_mov_b32 s72, 0", "={SGPR72}"() #0
  %sgpr73 = tail call i32 asm sideeffect "s_mov_b32 s73, 0", "={SGPR73}"() #0
  %sgpr74 = tail call i32 asm sideeffect "s_mov_b32 s74, 0", "={SGPR74}"() #0
  %sgpr75 = tail call i32 asm sideeffect "s_mov_b32 s75, 0", "={SGPR75}"() #0
  %sgpr76 = tail call i32 asm sideeffect "s_mov_b32 s76, 0", "={SGPR76}"() #0
  %sgpr77 = tail call i32 asm sideeffect "s_mov_b32 s77, 0", "={SGPR77}"() #0
  %sgpr78 = tail call i32 asm sideeffect "s_mov_b32 s78, 0", "={SGPR78}"() #0
  %sgpr79 = tail call i32 asm sideeffect "s_mov_b32 s79, 0", "={SGPR79}"() #0
  %sgpr80 = tail call i32 asm sideeffect "s_mov_b32 s80, 0", "={SGPR80}"() #0
  %sgpr81 = tail call i32 asm sideeffect "s_mov_b32 s81, 0", "={SGPR81}"() #0
  %sgpr82 = tail call i32 asm sideeffect "s_mov_b32 s82, 0", "={SGPR82}"() #0
  %sgpr83 = tail call i32 asm sideeffect "s_mov_b32 s83, 0", "={SGPR83}"() #0
  %sgpr84 = tail call i32 asm sideeffect "s_mov_b32 s84, 0", "={SGPR84}"() #0
  %sgpr85 = tail call i32 asm sideeffect "s_mov_b32 s85, 0", "={SGPR85}"() #0
  %sgpr86 = tail call i32 asm sideeffect "s_mov_b32 s86, 0", "={SGPR86}"() #0
  %sgpr87 = tail call i32 asm sideeffect "s_mov_b32 s87, 0", "={SGPR87}"() #0
  %sgpr88 = tail call i32 asm sideeffect "s_mov_b32 s88, 0", "={SGPR88}"() #0
  %sgpr89 = tail call i32 asm sideeffect "s_mov_b32 s89, 0", "={SGPR89}"() #0
  %sgpr90 = tail call i32 asm sideeffect "s_mov_b32 s90, 0", "={SGPR90}"() #0
  %sgpr91 = tail call i32 asm sideeffect "s_mov_b32 s91, 0", "={SGPR91}"() #0
  %sgpr92 = tail call i32 asm sideeffect "s_mov_b32 s92, 0", "={SGPR92}"() #0
  %sgpr93 = tail call i32 asm sideeffect "s_mov_b32 s93, 0", "={SGPR93}"() #0
  %sgpr94 = tail call i32 asm sideeffect "s_mov_b32 s94, 0", "={SGPR94}"() #0
  %sgpr95 = tail call i32 asm sideeffect "s_mov_b32 s95, 0", "={SGPR95}"() #0
  %sgpr96 = tail call i32 asm sideeffect "s_mov_b32 s96, 0", "={SGPR96}"() #0
  %sgpr97 = tail call i32 asm sideeffect "s_mov_b32 s97, 0", "={SGPR97}"() #0
  %sgpr98 = tail call i32 asm sideeffect "s_mov_b32 s98, 0", "={SGPR98}"() #0
  %sgpr99 = tail call i32 asm sideeffect "s_mov_b32 s99, 0", "={SGPR99}"() #0
  %sgpr100 = tail call i32 asm sideeffect "s_mov_b32 s100, 0", "={SGPR100}"() #0
  %sgpr101 = tail call i32 asm sideeffect "s_mov_b32 s101, 0", "={SGPR101}"() #0
  %sgpr102 = tail call i32 asm sideeffect "s_mov_b32 s102, 0", "={SGPR102}"() #0
  %sgpr103 = tail call i32 asm sideeffect "s_mov_b32 s103, 0", "={SGPR103}"() #0
  %vcc_lo = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={VCC_LO}"() #0
  %vcc_hi = tail call i32 asm sideeffect "s_mov_b32 $0, 0", "={VCC_HI}"() #0
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2 ; +8 dword branch

bb2: ; 28 bytes
  ; 24 byte asm
  call void asm sideeffect
   "v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64",""() #0
  br label %bb3

bb3:
  tail call void asm sideeffect "; reg use $0", "{SGPR0}"(i32 %sgpr0) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR1}"(i32 %sgpr1) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR2}"(i32 %sgpr2) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR3}"(i32 %sgpr3) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR4}"(i32 %sgpr4) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR5}"(i32 %sgpr5) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR6}"(i32 %sgpr6) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR7}"(i32 %sgpr7) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR8}"(i32 %sgpr8) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR9}"(i32 %sgpr9) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR10}"(i32 %sgpr10) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR11}"(i32 %sgpr11) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR12}"(i32 %sgpr12) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR13}"(i32 %sgpr13) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR14}"(i32 %sgpr14) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR15}"(i32 %sgpr15) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR16}"(i32 %sgpr16) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR17}"(i32 %sgpr17) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR18}"(i32 %sgpr18) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR19}"(i32 %sgpr19) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR20}"(i32 %sgpr20) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR21}"(i32 %sgpr21) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR22}"(i32 %sgpr22) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR23}"(i32 %sgpr23) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR24}"(i32 %sgpr24) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR25}"(i32 %sgpr25) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR26}"(i32 %sgpr26) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR27}"(i32 %sgpr27) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR28}"(i32 %sgpr28) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR29}"(i32 %sgpr29) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR30}"(i32 %sgpr30) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR31}"(i32 %sgpr31) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR32}"(i32 %sgpr32) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR33}"(i32 %sgpr33) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR34}"(i32 %sgpr34) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR35}"(i32 %sgpr35) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR36}"(i32 %sgpr36) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR37}"(i32 %sgpr37) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR38}"(i32 %sgpr38) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR39}"(i32 %sgpr39) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR40}"(i32 %sgpr40) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR41}"(i32 %sgpr41) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR42}"(i32 %sgpr42) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR43}"(i32 %sgpr43) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR44}"(i32 %sgpr44) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR45}"(i32 %sgpr45) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR46}"(i32 %sgpr46) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR47}"(i32 %sgpr47) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR48}"(i32 %sgpr48) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR49}"(i32 %sgpr49) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR50}"(i32 %sgpr50) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR51}"(i32 %sgpr51) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR52}"(i32 %sgpr52) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR53}"(i32 %sgpr53) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR54}"(i32 %sgpr54) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR55}"(i32 %sgpr55) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR56}"(i32 %sgpr56) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR57}"(i32 %sgpr57) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR58}"(i32 %sgpr58) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR59}"(i32 %sgpr59) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR60}"(i32 %sgpr60) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR61}"(i32 %sgpr61) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR62}"(i32 %sgpr62) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR63}"(i32 %sgpr63) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR64}"(i32 %sgpr64) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR65}"(i32 %sgpr65) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR66}"(i32 %sgpr66) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR67}"(i32 %sgpr67) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR68}"(i32 %sgpr68) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR69}"(i32 %sgpr69) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR70}"(i32 %sgpr70) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR71}"(i32 %sgpr71) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR72}"(i32 %sgpr72) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR73}"(i32 %sgpr73) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR74}"(i32 %sgpr74) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR75}"(i32 %sgpr75) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR76}"(i32 %sgpr76) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR77}"(i32 %sgpr77) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR78}"(i32 %sgpr78) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR79}"(i32 %sgpr79) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR80}"(i32 %sgpr80) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR81}"(i32 %sgpr81) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR82}"(i32 %sgpr82) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR83}"(i32 %sgpr83) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR84}"(i32 %sgpr84) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR85}"(i32 %sgpr85) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR86}"(i32 %sgpr86) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR87}"(i32 %sgpr87) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR88}"(i32 %sgpr88) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR89}"(i32 %sgpr89) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR90}"(i32 %sgpr90) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR91}"(i32 %sgpr91) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR92}"(i32 %sgpr92) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR93}"(i32 %sgpr93) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR94}"(i32 %sgpr94) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR95}"(i32 %sgpr95) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR96}"(i32 %sgpr96) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR97}"(i32 %sgpr97) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR98}"(i32 %sgpr98) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR99}"(i32 %sgpr99) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR100}"(i32 %sgpr100) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR101}"(i32 %sgpr101) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR102}"(i32 %sgpr102) #0
  tail call void asm sideeffect "; reg use $0", "{SGPR103}"(i32 %sgpr103) #0
  tail call void asm sideeffect "; reg use $0", "{VCC_LO}"(i32 %vcc_lo) #0
  tail call void asm sideeffect "; reg use $0", "{VCC_HI}"(i32 %vcc_hi) #0
  ret void
}

attributes #0 = { nounwind }
