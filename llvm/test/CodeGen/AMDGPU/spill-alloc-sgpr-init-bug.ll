; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck --check-prefix=TONGA %s

; On Tonga and Iceland, limited SGPR availability means care must be taken to
; allocate scratch registers correctly. Check that this test compiles without
; error.
; TONGA-LABEL: test
define amdgpu_kernel void @test(<256 x i32> addrspace(1)* %out, <256 x i32> addrspace(1)* %in) {
entry:
  %mbcnt.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %tid = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %mbcnt.lo)
  %aptr = getelementptr <256 x i32>, <256 x i32> addrspace(1)* %in, i32 %tid
  %a = load <256 x i32>, <256 x i32> addrspace(1)* %aptr
  call void asm sideeffect "", "~{memory}" ()
  %outptr = getelementptr <256 x i32>, <256 x i32> addrspace(1)* %in, i32 %tid
  store <256 x i32> %a, <256 x i32> addrspace(1)* %outptr

; mark 128-bit SGPR registers as used so they are unavailable for the
; scratch resource descriptor
  call void asm sideeffect "", "~{SGPR4},~{SGPR8},~{SGPR12},~{SGPR16},~{SGPR20},~{SGPR24},~{SGPR28}" ()
  call void asm sideeffect "", "~{SGPR32},~{SGPR36},~{SGPR40},~{SGPR44},~{SGPR48},~{SGPR52},~{SGPR56}" ()
  call void asm sideeffect "", "~{SGPR60},~{SGPR64},~{SGPR68}" ()
  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #0

attributes #0 = { nounwind readnone }
