; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck --check-prefix=TONGA %s

; On Tonga and Iceland, limited SGPR availability means care must be taken to
; allocate scratch registers correctly. Check that this test compiles without
; error.
; TONGA-LABEL: test
define amdgpu_kernel void @test(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %mbcnt.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %tid = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %mbcnt.lo)
  %aptr = getelementptr <256 x i32>, ptr addrspace(1) %in, i32 %tid
  %a = load <256 x i32>, ptr addrspace(1) %aptr
  call void asm sideeffect "", "~{memory}" ()
  %outptr = getelementptr <256 x i32>, ptr addrspace(1) %in, i32 %tid
  store <256 x i32> %a, ptr addrspace(1) %outptr

; mark 128-bit SGPR registers as used so they are unavailable for the
; scratch resource descriptor
  call void asm sideeffect "", "~{s4},~{s8},~{s12},~{s16},~{s20},~{s24},~{s28}" ()
  call void asm sideeffect "", "~{s32},~{s36},~{s40},~{s44},~{s48},~{s52},~{s56}" ()
  call void asm sideeffect "", "~{s60},~{s64},~{s68}" ()
  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #0

attributes #0 = { nounwind readnone }
