; RUN: llc -march=amdgcn -mcpu=verde -enable-misched=0 -post-RA-scheduler=0 < %s | FileCheck %s
; RUN: llc -regalloc=basic -march=amdgcn -mcpu=tonga -enable-misched=0 -post-RA-scheduler=0 < %s | FileCheck %s
 ;
; There is something about Tonga that causes this test to spend a lot of time
; in the default register allocator.


; When the offset of VGPR spills into scratch space gets too large, an additional SGPR
; is used to calculate the scratch load/store address. Make sure that this
; mechanism works even when many spills happen.

; Just test that it compiles successfully.
; CHECK-LABEL: test
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

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #1

attributes #1 = { nounwind readnone }
