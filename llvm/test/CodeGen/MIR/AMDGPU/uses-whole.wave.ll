; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1100 -stop-after=finalize-isel < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1100 -stop-after=finalize-isel < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: name: wwm
; GCN: usesWholeWave: true
define void @wwm(ptr addrspace(1) inreg %p) {
  %val = load i32, ptr addrspace(1) %p
  %wwm = tail call i32 @llvm.amdgcn.wwm.i32(i32 %val)
  store i32 %wwm, ptr addrspace(1) %p
  ret void
}

; GCN-LABEL: name: strict_wwm
; GCN: usesWholeWave: true
define void @strict_wwm(ptr addrspace(1) inreg %p) {
  %val = load i32, ptr addrspace(1) %p
  %wwm = tail call i32 @llvm.amdgcn.strict.wwm.i32(i32 %val)
  store i32 %wwm, ptr addrspace(1) %p
  ret void
}
