; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare void @llvm.amdgcn.s.decperflevel(i32) #0

; GCN-LABEL: {{^}}test_s_decperflevel:
; GCN: s_decperflevel 0{{$}}
; GCN: s_decperflevel 1{{$}}
; GCN: s_decperflevel 2{{$}}
; GCN: s_decperflevel 3{{$}}
; GCN: s_decperflevel 4{{$}}
; GCN: s_decperflevel 5{{$}}
; GCN: s_decperflevel 6{{$}}
; GCN: s_decperflevel 7{{$}}
; GCN: s_decperflevel 8{{$}}
; GCN: s_decperflevel 9{{$}}
; GCN: s_decperflevel 10{{$}}
; GCN: s_decperflevel 11{{$}}
; GCN: s_decperflevel 12{{$}}
; GCN: s_decperflevel 13{{$}}
; GCN: s_decperflevel 14{{$}}
; GCN: s_decperflevel 15{{$}}
define void @test_s_decperflevel(i32 %x) #0 {
  call void @llvm.amdgcn.s.decperflevel(i32 0)
  call void @llvm.amdgcn.s.decperflevel(i32 1)
  call void @llvm.amdgcn.s.decperflevel(i32 2)
  call void @llvm.amdgcn.s.decperflevel(i32 3)
  call void @llvm.amdgcn.s.decperflevel(i32 4)
  call void @llvm.amdgcn.s.decperflevel(i32 5)
  call void @llvm.amdgcn.s.decperflevel(i32 6)
  call void @llvm.amdgcn.s.decperflevel(i32 7)
  call void @llvm.amdgcn.s.decperflevel(i32 8)
  call void @llvm.amdgcn.s.decperflevel(i32 9)
  call void @llvm.amdgcn.s.decperflevel(i32 10)
  call void @llvm.amdgcn.s.decperflevel(i32 11)
  call void @llvm.amdgcn.s.decperflevel(i32 12)
  call void @llvm.amdgcn.s.decperflevel(i32 13)
  call void @llvm.amdgcn.s.decperflevel(i32 14)
  call void @llvm.amdgcn.s.decperflevel(i32 15)
  ret void
}

attributes #0 = { nounwind }
