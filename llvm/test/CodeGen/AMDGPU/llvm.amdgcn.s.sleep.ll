; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare void @llvm.amdgcn.s.sleep(i32) #0

; GCN-LABEL: {{^}}test_s_sleep:
; GCN: s_sleep 0{{$}}
; GCN: s_sleep 1{{$}}
; GCN: s_sleep 2{{$}}
; GCN: s_sleep 3{{$}}
; GCN: s_sleep 4{{$}}
; GCN: s_sleep 5{{$}}
; GCN: s_sleep 6{{$}}
; GCN: s_sleep 7{{$}}
; GCN: s_sleep 8{{$}}
; GCN: s_sleep 9{{$}}
; GCN: s_sleep 10{{$}}
; GCN: s_sleep 11{{$}}
; GCN: s_sleep 12{{$}}
; GCN: s_sleep 13{{$}}
; GCN: s_sleep 14{{$}}
; GCN: s_sleep 15{{$}}
define amdgpu_kernel void @test_s_sleep(i32 %x) #0 {
  call void @llvm.amdgcn.s.sleep(i32 0)
  call void @llvm.amdgcn.s.sleep(i32 1)
  call void @llvm.amdgcn.s.sleep(i32 2)
  call void @llvm.amdgcn.s.sleep(i32 3)
  call void @llvm.amdgcn.s.sleep(i32 4)
  call void @llvm.amdgcn.s.sleep(i32 5)
  call void @llvm.amdgcn.s.sleep(i32 6)
  call void @llvm.amdgcn.s.sleep(i32 7)

  ; Values that might only work on VI
  call void @llvm.amdgcn.s.sleep(i32 8)
  call void @llvm.amdgcn.s.sleep(i32 9)
  call void @llvm.amdgcn.s.sleep(i32 10)
  call void @llvm.amdgcn.s.sleep(i32 11)
  call void @llvm.amdgcn.s.sleep(i32 12)
  call void @llvm.amdgcn.s.sleep(i32 13)
  call void @llvm.amdgcn.s.sleep(i32 14)
  call void @llvm.amdgcn.s.sleep(i32 15)
  ret void
}

attributes #0 = { nounwind }
