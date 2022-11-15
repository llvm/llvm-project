; RUN: llc -march=amdgcn -mcpu=verde -enable-misched=0 -post-RA-scheduler=0 -amdgpu-spill-sgpr-to-vgpr=0 < %s | FileCheck -check-prefixes=CHECK,GFX6 %s
; RUN: llc -sgpr-regalloc=basic -vgpr-regalloc=basic -march=amdgcn -mcpu=tonga -enable-misched=0 -post-RA-scheduler=0 -amdgpu-spill-sgpr-to-vgpr=0 < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -march=amdgcn -mattr=-xnack,+enable-flat-scratch -mcpu=gfx900 -enable-misched=0 -post-RA-scheduler=0 -amdgpu-spill-sgpr-to-vgpr=0 < %s | FileCheck -check-prefixes=CHECK,GFX9-FLATSCR,FLATSCR %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -enable-misched=0 -post-RA-scheduler=0 -amdgpu-spill-sgpr-to-vgpr=0 -mattr=+enable-flat-scratch < %s | FileCheck -check-prefixes=CHECK,GFX10-FLATSCR,FLATSCR %s
;
; There is something about Tonga that causes this test to spend a lot of time
; in the default register allocator.


; When the offset of VGPR spills into scratch space gets too large, an additional SGPR
; is used to calculate the scratch load/store address. Make sure that this
; mechanism works even when many spills happen.

; Just test that it compiles successfully.
; CHECK-LABEL: test

; GFX9-FLATSCR: s_mov_b32 [[SOFF1:s[0-9]+]], 4{{$}}
; GFX9-FLATSCR: scratch_store_dwordx4 off, v[{{[0-9:]+}}], [[SOFF1]] ; 16-byte Folded Spill
; GFX9-FLATSCR: ;;#ASMSTART
; GFX9-FLATSCR: s_movk_i32 [[SOFF2:s[0-9]+]], 0x1{{[0-9a-f]+}}{{$}}
; GFX9-FLATSCR: scratch_load_dwordx4 v[{{[0-9:]+}}], off, [[SOFF2]] ; 16-byte Folded Reload

; GFX10-FLATSCR: scratch_store_dwordx4 off, v[{{[0-9:]+}}], off offset:{{[0-9]+}} ; 16-byte Folded Spill
; GFX10-FLATSCR: scratch_load_dwordx4 v[{{[0-9:]+}}], off, off offset:{{[0-9]+}} ; 16-byte Folded Reload
define amdgpu_kernel void @test(<1280 x i32> addrspace(1)* %out, <1280 x i32> addrspace(1)* %in) {
entry:
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %tid = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)

  %aptr = getelementptr <1280 x i32>, <1280 x i32> addrspace(1)* %in, i32 %tid
  %a = load <1280 x i32>, <1280 x i32> addrspace(1)* %aptr

; mark most VGPR registers as used to increase register pressure
  call void asm sideeffect "", "~{v4},~{v8},~{v12},~{v16},~{v20},~{v24},~{v28},~{v32}" ()
  call void asm sideeffect "", "~{v36},~{v40},~{v44},~{v48},~{v52},~{v56},~{v60},~{v64}" ()
  call void asm sideeffect "", "~{v68},~{v72},~{v76},~{v80},~{v84},~{v88},~{v92},~{v96}" ()
  call void asm sideeffect "", "~{v100},~{v104},~{v108},~{v112},~{v116},~{v120},~{v124},~{v128}" ()
  call void asm sideeffect "", "~{v132},~{v136},~{v140},~{v144},~{v148},~{v152},~{v156},~{v160}" ()
  call void asm sideeffect "", "~{v164},~{v168},~{v172},~{v176},~{v180},~{v184},~{v188},~{v192}" ()
  call void asm sideeffect "", "~{v196},~{v200},~{v204},~{v208},~{v212},~{v216},~{v220},~{v224}" ()

  %outptr = getelementptr <1280 x i32>, <1280 x i32> addrspace(1)* %out, i32 %tid
  store <1280 x i32> %a, <1280 x i32> addrspace(1)* %outptr

  ret void
}

; CHECK-LABEL: test_limited_sgpr
; GFX6: %bb.1:
; GFX6: s_mov_b64 exec, 0xff
; GFX6: buffer_store_dword [[SPILL_REG_0:v[0-9]+]]
; GFX6-COUNT-8: v_writelane_b32 [[SPILL_REG_0]]
; GFX6: v_mov_b32_e32 [[OFFSET_REG0:v[0-9]+]], 0x[[OFFSET0:[0-9a-f]+]]
; GFX6: buffer_store_dword [[SPILL_REG_0]], [[OFFSET_REG0]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX6: buffer_load_dword [[SPILL_REG_0]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GFX6: s_mov_b64 exec, s


; GFX6: s_mov_b64 exec, 0xff
; GFX6: v_mov_b32_e32 [[RELOAD_OFFSET_REG0:v[0-9]+]], 0x[[RELOAD_OFFSET0:[0-9a-f]+]]
; GFX6: buffer_store_dword [[RELOAD_REG_0:v[0-9]+]], off,
; GFX6: buffer_load_dword [[RELOAD_REG_0]], [[RELOAD_OFFSET_REG0]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX6-COUNT-8: v_readlane_b32 s{{[0-9]+}}, [[RELOAD_REG_0]]
; GFX6: buffer_load_dword [[RELOAD_REG_0]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GFX6: s_mov_b64 exec,


; GFX6: s_mov_b64 exec, 0xff
; GFX6: buffer_store_dword [[SPILL_REG_1:v[0-9]+]]
; GFX6-COUNT-8: v_writelane_b32 [[SPILL_REG_1]]
; GFX6: v_mov_b32_e32 [[OFFSET_REG1:v[0-9]+]], 0x[[OFFSET1:[0-9a-f]+]]
; GFX6: buffer_store_dword [[SPILL_REG_1]], [[OFFSET_REG1]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX6: buffer_load_dword [[SPILL_REG_1]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GFX6: s_mov_b64 exec, s


; GFX6: s_mov_b64 exec, 0xff
; GFX6: v_mov_b32_e32 [[RELOAD_OFFSET_REG1:v[0-9]+]], 0x[[RELOAD_OFFSET1:[0-9a-f]+]]
; GFX6: buffer_store_dword [[RELOAD_REG_1:v[0-9]+]], off,
; GFX6: buffer_load_dword [[RELOAD_REG_1]], [[RELOAD_OFFSET_REG1]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX6-COUNT-8: v_readlane_b32 s{{[0-9]+}}, [[RELOAD_REG_1]]
; GFX6: buffer_load_dword [[RELOAD_REG_1]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GFX6: s_mov_b64 exec,


; GFX6: s_mov_b64 exec, 0xff
; GFX6: buffer_store_dword [[SPILL_REG_2:v[0-9]+]]
; GFX6-COUNT-8: v_writelane_b32 [[SPILL_REG_2]]
; GFX6: v_mov_b32_e32 [[OFFSET_REG2:v[0-9]+]], 0x[[OFFSET2:[0-9a-f]+]]
; GFX6: buffer_store_dword [[SPILL_REG_2]], [[OFFSET_REG2]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX6: buffer_load_dword [[SPILL_REG_2]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GFX6: s_mov_b64 exec, s


; GFX6: s_mov_b64 exec, 0xff
; GFX6: buffer_store_dword [[SPILL_REG_3:v[0-9]+]]
; GFX6-COUNT-8: v_writelane_b32 [[SPILL_REG_3]]
; GFX6: v_mov_b32_e32 [[OFFSET_REG3:v[0-9]+]], 0x[[OFFSET3:[0-9a-f]+]]
; GFX6: buffer_store_dword [[SPILL_REG_3]], [[OFFSET_REG3]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX6: buffer_load_dword [[SPILL_REG_3]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GFX6: s_mov_b64 exec, s


; GFX6: s_mov_b64 exec, 0xff
; GFX6: buffer_store_dword [[SPILL_REG_4:v[0-9]+]]
; GFX6-COUNT-4: v_writelane_b32 [[SPILL_REG_4]]
; GFX6: v_mov_b32_e32 [[OFFSET_REG4:v[0-9]+]], 0x[[OFFSET4:[0-9a-f]+]]
; GFX6: buffer_store_dword [[SPILL_REG_4]], [[OFFSET_REG4]], s{{\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX6: buffer_load_dword [[SPILL_REG_4]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GFX6: s_mov_b64 exec, s

; GFX6: NumSgprs: 48
; GFX6: ScratchSize: 8608

; FLATSCR:           s_movk_i32 [[SOFF1:s[0-9]+]], 0x
; GFX9-FLATSCR:      s_waitcnt vmcnt(0)
; FLATSCR:           scratch_store_dwordx4 off, v[{{[0-9:]+}}], [[SOFF1]] ; 16-byte Folded Spill
; FLATSCR:           s_movk_i32 [[SOFF2:s[0-9]+]], 0x
; FLATSCR:           scratch_load_dwordx4 v[{{[0-9:]+}}], off, [[SOFF2]] ; 16-byte Folded Reload
define amdgpu_kernel void @test_limited_sgpr(<64 x i32> addrspace(1)* %out, <64 x i32> addrspace(1)* %in) #0 {
entry:
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %tid = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)

; allocate enough scratch to go beyond 2^12 addressing
  %scratch = alloca <1280 x i32>, align 8, addrspace(5)

; load VGPR data
  %aptr = getelementptr <64 x i32>, <64 x i32> addrspace(1)* %in, i32 %tid
  %a = load <64 x i32>, <64 x i32> addrspace(1)* %aptr

; make sure scratch is used
  %x = extractelement <64 x i32> %a, i32 0
  %sptr0 = getelementptr <1280 x i32>, <1280 x i32> addrspace(5)* %scratch, i32 %x, i32 0
  store i32 1, i32 addrspace(5)* %sptr0

; fill up SGPRs
  %sgpr0 = call <8 x i32> asm sideeffect "; def $0", "=s" ()
  %sgpr1 = call <8 x i32> asm sideeffect "; def $0", "=s" ()
  %sgpr2 = call <8 x i32> asm sideeffect "; def $0", "=s" ()
  %sgpr3 = call <8 x i32> asm sideeffect "; def $0", "=s" ()
  %sgpr4 = call <4 x i32> asm sideeffect "; def $0", "=s" ()
  %sgpr5 = call <2 x i32> asm sideeffect "; def $0", "=s" ()
  %sgpr6 = call <2 x i32> asm sideeffect "; def $0", "=s" ()
  %sgpr7 = call i32 asm sideeffect "; def $0", "=s" ()

  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %bb0, label %ret

bb0:
; create SGPR pressure
  call void asm sideeffect "; use $0,$1,$2,$3,$4,$5,$6", "s,s,s,s,s,s,s,s"(<8 x i32> %sgpr0, <8 x i32> %sgpr1, <8 x i32> %sgpr2, <8 x i32> %sgpr3, <4 x i32> %sgpr4, <2 x i32> %sgpr5, <2 x i32> %sgpr6, i32 %sgpr7)

; mark most VGPR registers as used to increase register pressure
  call void asm sideeffect "", "~{v4},~{v8},~{v12},~{v16},~{v20},~{v24},~{v28},~{v32}" ()
  call void asm sideeffect "", "~{v36},~{v40},~{v44},~{v48},~{v52},~{v56},~{v60},~{v64}" ()
  call void asm sideeffect "", "~{v68},~{v72},~{v76},~{v80},~{v84},~{v88},~{v92},~{v96}" ()
  call void asm sideeffect "", "~{v100},~{v104},~{v108},~{v112},~{v116},~{v120},~{v124},~{v128}" ()
  call void asm sideeffect "", "~{v132},~{v136},~{v140},~{v144},~{v148},~{v152},~{v156},~{v160}" ()
  call void asm sideeffect "", "~{v164},~{v168},~{v172},~{v176},~{v180},~{v184},~{v188},~{v192}" ()
  call void asm sideeffect "", "~{v196},~{v200},~{v204},~{v208},~{v212},~{v216},~{v220},~{v224}" ()
  br label %ret

ret:
  %outptr = getelementptr <64 x i32>, <64 x i32> addrspace(1)* %out, i32 %tid
  store <64 x i32> %a, <64 x i32> addrspace(1)* %outptr

  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #1

attributes #0 = { "amdgpu-waves-per-eu"="10,10" }
attributes #1 = { nounwind readnone }
