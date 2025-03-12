; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+xnack -amdgpu-use-amdgpu-trackers=1  2>&1  < %s | FileCheck -check-prefixes=ERR-GCNTRACKERS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+xnack 2>&1  < %s | FileCheck -check-prefixes=GCN %s

%asm.output = type { <16 x i32>, <16 x i32>, <16 x i32>, <8 x i32>, <2 x i32>, i32, ; sgprs
                     <16 x i32>, <7 x i32>, ; vgprs
                     i64 ; vcc
                     }

%asm.output2 = type { <16 x i32>, <16 x i32>, <16 x i32>, <8 x i32>, <2 x i32>, i32, ; sgprs
                     <16 x i32>, <5 x i32>, ; vgprs
                     i64 ; vcc
                     }

%asm.output3 = type { <16 x i32>, <16 x i32>, <16 x i32>, <8 x i32>, <2 x i32>, ; sgprs
                     <16 x i32>, <6 x i32>, ; vgprs
                     i64 ; vcc
                     }

; ERR-GCNTRACKERS: ran out of registers during register allocation
; GCN-NOT: ran out of registers during register allocation

; FIXME: GCN Trackers do not track pressure from PhysRegs, so scheduling is actually worse

define void @scalar_mov_materializes_frame_index_no_live_scc_no_live_sgprs() #0 {
  %alloca0 = alloca [4096 x i32], align 64, addrspace(5)
  %alloca1 = alloca i32, align 4, addrspace(5)
  call void asm sideeffect "; use alloca0 $0", "v"(ptr addrspace(5) %alloca0)

  %asm = call %asm.output asm sideeffect
    "; def $0, $1, $2, $3, $4, $5, $6, $7, $8",
    "={s[0:15]},={s[16:31]},={s[32:47]},={s[48:55]},={s[56:57]},={s58},={v[0:15]},={v[16:22]},={vcc}"()

  %s0 = extractvalue %asm.output %asm, 0
  %s1 = extractvalue %asm.output %asm, 1
  %s2 = extractvalue %asm.output %asm, 2
  %s3 = extractvalue %asm.output %asm, 3
  %s4 = extractvalue %asm.output %asm, 4
  %s5 = extractvalue %asm.output %asm, 5

  %v0 = extractvalue %asm.output %asm, 6
  %v1 = extractvalue %asm.output %asm, 7

  %vcc = extractvalue %asm.output %asm, 8

  ; scc is unavailable since it is live in
  call void asm sideeffect "; use $0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10",
                           "{s[0:15]},{s[16:31]},{s[32:47]},{s[48:55]},{s[56:57]},{s58},{v[0:15]},{v[16:22]},{vcc},{s59},{scc}"(
    <16 x i32> %s0,
    <16 x i32> %s1,
    <16 x i32> %s2,
    <8 x i32> %s3,
    <2 x i32> %s4,
    i32 %s5,
    <16 x i32> %v0,
    <7 x i32> %v1,
    i64 %vcc,
    ptr addrspace(5) %alloca1,
    i32 0) ; use of scc

  ret void
}

attributes #0 = { nounwind alignstack=64 "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="10,10" "no-realign-stack" }
attributes #1 = { nounwind alignstack=16 "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="10,10" "no-realign-stack" }

