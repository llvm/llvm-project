; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1100 -stop-after=finalize-isel < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1100 -stop-after=finalize-isel < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: name: init_wwm
; GCN: hasInitWholeWave: true
define void @init_wwm(ptr addrspace(1) inreg %p) {
entry:
  %entry_exec = call i1 @llvm.amdgcn.init.whole.wave()
  br i1 %entry_exec, label %bb.1, label %bb.2

bb.1:
  store i32 1, ptr addrspace(1) %p
  br label %bb.2

bb.2:
  ret void
}
