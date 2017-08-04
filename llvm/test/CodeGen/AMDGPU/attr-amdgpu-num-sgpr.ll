; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -amdgpu-spill-sgpr-to-smem=0 -verify-machineinstrs < %s | FileCheck -check-prefix=TOSGPR -check-prefix=ALL %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -amdgpu-spill-sgpr-to-smem=1 -verify-machineinstrs < %s | FileCheck -check-prefix=TOSMEM -check-prefix=ALL %s

; If spilling to smem, additional registers are used for the resource
; descriptor.

; ALL-LABEL: {{^}}max_9_sgprs:

; ALL: SGPRBlocks: 1
; ALL: NumSGPRsForWavesPerEU: 9
define amdgpu_kernel void @max_9_sgprs(i32 addrspace(1)* %out1,

                          i32 addrspace(1)* %out2,
                          i32 addrspace(1)* %out3,
                          i32 addrspace(1)* %out4,
                          i32 addrspace(1)* %out5,
                          i32 %one, i32 %two, i32 %three, i32 %four, i32 %five) #0 {
  store i32 %one, i32 addrspace(1)* %out1
  store i32 %two, i32 addrspace(1)* %out2
  store i32 %three, i32 addrspace(1)* %out3
  store i32 %four, i32 addrspace(1)* %out4
  store i32 %five, i32 addrspace(1)* %out5
  ret void
}

; private resource: 4
; scratch wave offset: 1
; workgroup ids: 3
; dispatch id: 2
; queue ptr: 2
; flat scratch init: 2
; ---------------------
; total: 14

; + reserved vcc = 16

; Because we can't handle re-using the last few input registers as the
; special vcc etc. registers (as well as decide to not use the unused
; features when the number of registers is frozen), this ends up using
; more than expected.

; XALL-LABEL: {{^}}max_12_sgprs_14_input_sgprs:
; XTOSGPR: SGPRBlocks: 1
; XTOSGPR: NumSGPRsForWavesPerEU: 16

; XTOSMEM: s_mov_b64 s[10:11], s[2:3]
; XTOSMEM: s_mov_b64 s[8:9], s[0:1]
; XTOSMEM: s_mov_b32 s7, s13

; XTOSMEM: SGPRBlocks: 1
; XTOSMEM: NumSGPRsForWavesPerEU: 16
;
; This test case is disabled: When calculating the spillslot addresses AMDGPU
; creates an extra vreg to save/restore m0 which in a point of maximum register
; pressure would trigger an endless loop; the compiler aborts earlier with
; "Incomplete scavenging after 2nd pass" in practice.
;define amdgpu_kernel void @max_12_sgprs_14_input_sgprs(i32 addrspace(1)* %out1,
;                                        i32 addrspace(1)* %out2,
;                                        i32 addrspace(1)* %out3,
;                                        i32 addrspace(1)* %out4,
;                                        i32 %one, i32 %two, i32 %three, i32 %four) #2 {
;  %x.0 = call i32 @llvm.amdgcn.workgroup.id.x()
;  %x.1 = call i32 @llvm.amdgcn.workgroup.id.y()
;  %x.2 = call i32 @llvm.amdgcn.workgroup.id.z()
;  %x.3 = call i64 @llvm.amdgcn.dispatch.id()
;  %x.4 = call i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
;  %x.5 = call i8 addrspace(2)* @llvm.amdgcn.queue.ptr()
;  store volatile i32 0, i32* undef
;  br label %stores
;
;stores:
;  store volatile i32 %x.0, i32 addrspace(1)* undef
;  store volatile i32 %x.0, i32 addrspace(1)* undef
;  store volatile i32 %x.0, i32 addrspace(1)* undef
;  store volatile i64 %x.3, i64 addrspace(1)* undef
;  store volatile i8 addrspace(2)* %x.4, i8 addrspace(2)* addrspace(1)* undef
;  store volatile i8 addrspace(2)* %x.5, i8 addrspace(2)* addrspace(1)* undef
;
;  store i32 %one, i32 addrspace(1)* %out1
;  store i32 %two, i32 addrspace(1)* %out2
;  store i32 %three, i32 addrspace(1)* %out3
;  store i32 %four, i32 addrspace(1)* %out4
;  ret void
;}

; The following test is commented out for now; http://llvm.org/PR31230
; XALL-LABEL: max_12_sgprs_12_input_sgprs{{$}}
; ; Make sure copies for input buffer are not clobbered. This requires
; ; swapping the order the registers are copied from what normally
; ; happens.

; XTOSMEM: s_mov_b32 s5, s11
; XTOSMEM: s_add_u32 m0, s5,
; XTOSMEM: s_buffer_store_dword vcc_lo, s[0:3], m0

; XALL: SGPRBlocks: 2
; XALL: NumSGPRsForWavesPerEU: 18
;define amdgpu_kernel void @max_12_sgprs_12_input_sgprs(i32 addrspace(1)* %out1,
;                                        i32 addrspace(1)* %out2,
;                                        i32 addrspace(1)* %out3,
;                                        i32 addrspace(1)* %out4,
;                                        i32 %one, i32 %two, i32 %three, i32 %four) #2 {
;  store volatile i32 0, i32* undef
;  %x.0 = call i32 @llvm.amdgcn.workgroup.id.x()
;  store volatile i32 %x.0, i32 addrspace(1)* undef
;  %x.1 = call i32 @llvm.amdgcn.workgroup.id.y()
;  store volatile i32 %x.0, i32 addrspace(1)* undef
;  %x.2 = call i32 @llvm.amdgcn.workgroup.id.z()
;  store volatile i32 %x.0, i32 addrspace(1)* undef
;  %x.3 = call i64 @llvm.amdgcn.dispatch.id()
;  store volatile i64 %x.3, i64 addrspace(1)* undef
;  %x.4 = call i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
;  store volatile i8 addrspace(2)* %x.4, i8 addrspace(2)* addrspace(1)* undef
;
;  store i32 %one, i32 addrspace(1)* %out1
;  store i32 %two, i32 addrspace(1)* %out2
;  store i32 %three, i32 addrspace(1)* %out3
;  store i32 %four, i32 addrspace(1)* %out4
;  ret void
;}

declare i32 @llvm.amdgcn.workgroup.id.x() #1
declare i32 @llvm.amdgcn.workgroup.id.y() #1
declare i32 @llvm.amdgcn.workgroup.id.z() #1
declare i64 @llvm.amdgcn.dispatch.id() #1
declare i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #1
declare i8 addrspace(2)* @llvm.amdgcn.queue.ptr() #1

attributes #0 = { nounwind "amdgpu-num-sgpr"="14" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "amdgpu-num-sgpr"="12" }
attributes #3 = { nounwind "amdgpu-num-sgpr"="11" }
