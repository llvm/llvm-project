; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}max_14_sgprs:

; FIXME: Should be ablo to skip this copying of the private segment
; buffer because all the SGPR spills are to VGPRs.

; CHECK: s_mov_b64 s[6:7], s[2:3]
; CHECK: s_mov_b64 s[4:5], s[0:1]

; CHECK: SGPRBlocks: 1
; CHECK: NumSGPRsForWavesPerEU: 14
define void @max_14_sgprs(i32 addrspace(1)* %out1,
                          i32 addrspace(1)* %out2,
                          i32 addrspace(1)* %out3,
                          i32 addrspace(1)* %out4,
                          i32 %one, i32 %two, i32 %three, i32 %four) #0 {
  store i32 %one, i32 addrspace(1)* %out1
  store i32 %two, i32 addrspace(1)* %out2
  store i32 %three, i32 addrspace(1)* %out3
  store i32 %four, i32 addrspace(1)* %out4
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

; + reserved vcc, flat_scratch = 18

; Because we can't handle re-using the last few input registers as the
; special vcc etc. registers (as well as decide to not use the unused
; features when the number of registers is frozen), this ends up using
; more than expected.

; ALL-LABEL: {{^}}max_12_sgprs_14_input_sgprs:
; TOSGPR: SGPRBlocks: 2
; TOSGPR: NumSGPRsForWavesPerEU: 18

; TOSMEM: s_mov_b64 s[6:7], s[2:3]
; TOSMEM: s_mov_b32 s9, s13
; TOSMEM: s_mov_b64 s[4:5], s[0:1]

; TOSMEM: SGPRBlocks: 2
; TOSMEM: NumSGPRsForWavesPerEU: 18
define void @max_12_sgprs_14_input_sgprs(i32 addrspace(1)* %out1,
                                        i32 addrspace(1)* %out2,
                                        i32 addrspace(1)* %out3,
                                        i32 addrspace(1)* %out4,
                                        i32 %one, i32 %two, i32 %three, i32 %four) #2 {
  store volatile i32 0, i32* undef
  %x.0 = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %x.0, i32 addrspace(1)* undef
  %x.1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %x.0, i32 addrspace(1)* undef
  %x.2 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %x.0, i32 addrspace(1)* undef
  %x.3 = call i64 @llvm.amdgcn.dispatch.id()
  store volatile i64 %x.3, i64 addrspace(1)* undef
  %x.4 = call i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  store volatile i8 addrspace(2)* %x.4, i8 addrspace(2)* addrspace(1)* undef
  %x.5 = call i8 addrspace(2)* @llvm.amdgcn.queue.ptr()
  store volatile i8 addrspace(2)* %x.5, i8 addrspace(2)* addrspace(1)* undef

  store i32 %one, i32 addrspace(1)* %out1
  store i32 %two, i32 addrspace(1)* %out2
  store i32 %three, i32 addrspace(1)* %out3
  store i32 %four, i32 addrspace(1)* %out4
  ret void
}

; ALL-LABEL: max_12_sgprs_12_input_sgprs{{$}}
; ; Make sure copies for input buffer are not clobbered. This requires
; ; swapping the order the registers are copied from what normally
; ; happens.

; TOSMEM: s_mov_b64 s[6:7], s[2:3]
; TOSMEM: s_mov_b64 s[4:5], s[0:1]
; TOSMEM: s_mov_b32 s3, s11

; ALL: SGPRBlocks: 1
; ALL: NumSGPRsForWavesPerEU: 16
define void @max_12_sgprs_12_input_sgprs(i32 addrspace(1)* %out1,
                                        i32 addrspace(1)* %out2,
                                        i32 addrspace(1)* %out3,
                                        i32 addrspace(1)* %out4,
                                        i32 %one, i32 %two, i32 %three, i32 %four) #2 {
  store volatile i32 0, i32* undef
  %x.0 = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %x.0, i32 addrspace(1)* undef
  %x.1 = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %x.0, i32 addrspace(1)* undef
  %x.2 = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %x.0, i32 addrspace(1)* undef
  %x.3 = call i64 @llvm.amdgcn.dispatch.id()
  store volatile i64 %x.3, i64 addrspace(1)* undef
  %x.4 = call i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  store volatile i8 addrspace(2)* %x.4, i8 addrspace(2)* addrspace(1)* undef

  store i32 %one, i32 addrspace(1)* %out1
  store i32 %two, i32 addrspace(1)* %out2
  store i32 %three, i32 addrspace(1)* %out3
  store i32 %four, i32 addrspace(1)* %out4
  ret void
}

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
