; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<divergence>' -disable-output %s 2>&1 | FileCheck %s

; CHECK-LABEL: for function 'readfirstlane':
define amdgpu_kernel void @readfirstlane() {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT:  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %first.lane = call i32 @llvm.amdgcn.readfirstlane(i32 %id.x)
; CHECK-NOT: DIVERGENT:  %first.lane = call i32 @llvm.amdgcn.readfirstlane(i32 %id.x)
  ret void
}

; CHECK-LABEL: for function 'icmp':
define amdgpu_kernel void @icmp(i32 inreg %x) {
; CHECK-NOT: DIVERGENT:  %icmp = call i64 @llvm.amdgcn.icmp.i32
  %icmp = call i64 @llvm.amdgcn.icmp.i32(i32 %x, i32 0, i32 33)
  ret void
}

; CHECK-LABEL: for function 'fcmp':
define amdgpu_kernel void @fcmp(float inreg %x, float inreg %y) {
; CHECK-NOT: DIVERGENT:  %fcmp = call i64 @llvm.amdgcn.fcmp.i32
  %fcmp = call i64 @llvm.amdgcn.fcmp.i32(float %x, float %y, i32 33)
  ret void
}

; CHECK-LABEL: for function 'ballot':
define amdgpu_kernel void @ballot(i1 inreg %x) {
; CHECK-NOT: DIVERGENT:  %ballot = call i64 @llvm.amdgcn.ballot.i32
  %ballot = call i64 @llvm.amdgcn.ballot.i32(i1 %x)
  ret void
}

; SGPR asm outputs are uniform regardless of the input operands.
; CHECK-LABEL: for function 'asm_sgpr':
; CHECK: DIVERGENT: i32 %divergent
; CHECK-NOT: DIVERGENT
define i32 @asm_sgpr(i32 %divergent) {
  %sgpr = call i32 asm "; def $0, $1","=s,v"(i32 %divergent)
  ret i32 %sgpr
}

; CHECK-LABEL: Divergence Analysis' for function 'asm_mixed_sgpr_vgpr':
; CHECK: DIVERGENT: %asm = call { i32, i32 } asm "; def $0, $1, $2", "=s,=v,v"(i32 %divergent)
; CHECK-NEXT: {{^[ \t]+}}%sgpr = extractvalue { i32, i32 } %asm, 0
; CHECK-NEXT: DIVERGENT:       %vgpr = extractvalue { i32, i32 } %asm, 1
define void @asm_mixed_sgpr_vgpr(i32 %divergent) {
  %asm = call { i32, i32 } asm "; def $0, $1, $2","=s,=v,v"(i32 %divergent)
  %sgpr = extractvalue { i32, i32 } %asm, 0
  %vgpr = extractvalue { i32, i32 } %asm, 1
  store i32 %sgpr, ptr addrspace(1) undef
  store i32 %vgpr, ptr addrspace(1) undef
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.readfirstlane(i32) #0
declare i64 @llvm.amdgcn.icmp.i32(i32, i32, i32) #1
declare i64 @llvm.amdgcn.fcmp.i32(float, float, i32) #1
declare i64 @llvm.amdgcn.ballot.i32(i1) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone convergent }
