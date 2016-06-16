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
define void @test(<1280 x i32> addrspace(1)* %out, <1280 x i32> addrspace(1)* %in) {
entry:
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %tid = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)

  %aptr = getelementptr <1280 x i32>, <1280 x i32> addrspace(1)* %in, i32 %tid
  %a = load <1280 x i32>, <1280 x i32> addrspace(1)* %aptr

; mark most VGPR registers as used to increase register pressure
  call void asm sideeffect "", "~{VGPR4},~{VGPR8},~{VGPR12},~{VGPR16},~{VGPR20},~{VGPR24},~{VGPR28},~{VGPR32}" ()
  call void asm sideeffect "", "~{VGPR36},~{VGPR40},~{VGPR44},~{VGPR48},~{VGPR52},~{VGPR56},~{VGPR60},~{VGPR64}" ()
  call void asm sideeffect "", "~{VGPR68},~{VGPR72},~{VGPR76},~{VGPR80},~{VGPR84},~{VGPR88},~{VGPR92},~{VGPR96}" ()
  call void asm sideeffect "", "~{VGPR100},~{VGPR104},~{VGPR108},~{VGPR112},~{VGPR116},~{VGPR120},~{VGPR124},~{VGPR128}" ()
  call void asm sideeffect "", "~{VGPR132},~{VGPR136},~{VGPR140},~{VGPR144},~{VGPR148},~{VGPR152},~{VGPR156},~{VGPR160}" ()
  call void asm sideeffect "", "~{VGPR164},~{VGPR168},~{VGPR172},~{VGPR176},~{VGPR180},~{VGPR184},~{VGPR188},~{VGPR192}" ()
  call void asm sideeffect "", "~{VGPR196},~{VGPR200},~{VGPR204},~{VGPR208},~{VGPR212},~{VGPR216},~{VGPR220},~{VGPR224}" ()

  %outptr = getelementptr <1280 x i32>, <1280 x i32> addrspace(1)* %out, i32 %tid
  store <1280 x i32> %a, <1280 x i32> addrspace(1)* %outptr

  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #1

attributes #1 = { nounwind readnone }
