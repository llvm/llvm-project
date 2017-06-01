; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare void @llvm.amdgcn.s.incperflevel(i32) #0

; GCN-LABEL: {{^}}test_s_incperflevel:
; GCN: s_incperflevel 0{{$}}
; GCN: s_incperflevel 1{{$}}
; GCN: s_incperflevel 2{{$}}
; GCN: s_incperflevel 3{{$}}
; GCN: s_incperflevel 4{{$}}
; GCN: s_incperflevel 5{{$}}
; GCN: s_incperflevel 6{{$}}
; GCN: s_incperflevel 7{{$}}
; GCN: s_incperflevel 8{{$}}
; GCN: s_incperflevel 9{{$}}
; GCN: s_incperflevel 10{{$}}
; GCN: s_incperflevel 11{{$}}
; GCN: s_incperflevel 12{{$}}
; GCN: s_incperflevel 13{{$}}
; GCN: s_incperflevel 14{{$}}
; GCN: s_incperflevel 15{{$}}
define void @test_s_incperflevel(i32 %x) #0 {
  call void @llvm.amdgcn.s.incperflevel(i32 0)
  call void @llvm.amdgcn.s.incperflevel(i32 1)
  call void @llvm.amdgcn.s.incperflevel(i32 2)
  call void @llvm.amdgcn.s.incperflevel(i32 3)
  call void @llvm.amdgcn.s.incperflevel(i32 4)
  call void @llvm.amdgcn.s.incperflevel(i32 5)
  call void @llvm.amdgcn.s.incperflevel(i32 6)
  call void @llvm.amdgcn.s.incperflevel(i32 7)
  call void @llvm.amdgcn.s.incperflevel(i32 8)
  call void @llvm.amdgcn.s.incperflevel(i32 9)
  call void @llvm.amdgcn.s.incperflevel(i32 10)
  call void @llvm.amdgcn.s.incperflevel(i32 11)
  call void @llvm.amdgcn.s.incperflevel(i32 12)
  call void @llvm.amdgcn.s.incperflevel(i32 13)
  call void @llvm.amdgcn.s.incperflevel(i32 14)
  call void @llvm.amdgcn.s.incperflevel(i32 15)
  ret void
}

attributes #0 = { nounwind }
