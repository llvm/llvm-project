; RUN: opt -mtriple amdgcn-mesa-mesa3d -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; Tests uniformity of the AMDGPU control flow intrinsics. Their lane-mask
; operands and results are i1 values that are divergent (per-lane).

; CHECK: for function 'test_if_break':
; CHECK: DIVERGENT: %cond = icmp eq i32 %arg0, 0
; CHECK: DIVERGENT: %break = call i1 @llvm.amdgcn.if.break(i1 %cond, i1 %saved)
define amdgpu_ps void @test_if_break(i32 %arg0, i1 inreg %saved) {
entry:
  %cond = icmp eq i32 %arg0, 0
  %break = call i1 @llvm.amdgcn.if.break(i1 %cond, i1 %saved)
  store volatile i1 %break, ptr addrspace(1) undef
  ret void
}

; CHECK: for function 'test_if':
; CHECK: DIVERGENT: %cond = icmp eq i32 %arg0, 0
; CHECK-NEXT: DIVERGENT: %if = call { i1, i1 } @llvm.amdgcn.if(i1 %cond)
; CHECK-NEXT: DIVERGENT: %if.bool = extractvalue { i1, i1 } %if, 0
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: %if.bool.ext = zext i1 %if.bool to i32
define void @test_if(i32 %arg0) {
entry:
  %cond = icmp eq i32 %arg0, 0
  %if = call { i1, i1 } @llvm.amdgcn.if(i1 %cond)
  %if.bool = extractvalue { i1, i1 } %if, 0
  %if.mask = extractvalue { i1, i1 } %if, 1
  %if.bool.ext = zext i1 %if.bool to i32
  store volatile i32 %if.bool.ext, ptr addrspace(1) undef
  store volatile i1 %if.mask, ptr addrspace(1) undef
  ret void
}

; The result should still be treated as divergent, even with a uniform source.
; CHECK: for function 'test_if_uniform':
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: %if = call { i1, i1 } @llvm.amdgcn.if(i1 %cond)
; CHECK-NEXT: DIVERGENT: %if.bool = extractvalue { i1, i1 } %if, 0
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: %if.bool.ext = zext i1 %if.bool to i32
define amdgpu_ps void @test_if_uniform(i32 inreg %arg0) {
entry:
  %cond = icmp eq i32 %arg0, 0
  %if = call { i1, i1 } @llvm.amdgcn.if(i1 %cond)
  %if.bool = extractvalue { i1, i1 } %if, 0
  %if.mask = extractvalue { i1, i1 } %if, 1
  %if.bool.ext = zext i1 %if.bool to i32
  store volatile i32 %if.bool.ext, ptr addrspace(1) undef
  store volatile i1 %if.mask, ptr addrspace(1) undef
  ret void
}

; CHECK: for function 'test_loop_uniform':
; CHECK: DIVERGENT: %loop = call i1 @llvm.amdgcn.loop(i1 %mask)
define amdgpu_ps void @test_loop_uniform(i1 inreg %mask) {
entry:
  %loop = call i1 @llvm.amdgcn.loop(i1 %mask)
  %loop.ext = zext i1 %loop to i32
  store volatile i32 %loop.ext, ptr addrspace(1) undef
  ret void
}

; CHECK: for function 'test_else':
; CHECK: DIVERGENT: %else = call { i1, i1 } @llvm.amdgcn.else(i1 %mask)
; CHECK: DIVERGENT:       %else.bool = extractvalue { i1, i1 } %else, 0
; CHECK: {{^[ \t]+}}%else.mask = extractvalue { i1, i1 } %else, 1
define amdgpu_ps void @test_else(i1 inreg %mask) {
entry:
  %else = call { i1, i1 } @llvm.amdgcn.else(i1 %mask)
  %else.bool = extractvalue { i1, i1 } %else, 0
  %else.mask = extractvalue { i1, i1 } %else, 1
  %else.bool.ext = zext i1 %else.bool to i32
  store volatile i32 %else.bool.ext, ptr addrspace(1) undef
  store volatile i1 %else.mask, ptr addrspace(1) undef
  ret void
}

; This case is probably always broken
; CHECK: for function 'test_else_divergent_mask':
; CHECK: DIVERGENT: %if = call { i1, i1 } @llvm.amdgcn.else(i1 %mask)
; CHECK-NEXT: DIVERGENT: %if.bool = extractvalue { i1, i1 } %if, 0
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: %if.bool.ext = zext i1 %if.bool to i32
define void @test_else_divergent_mask(i1 %mask) {
entry:
  %if = call { i1, i1 } @llvm.amdgcn.else(i1 %mask)
  %if.bool = extractvalue { i1, i1 } %if, 0
  %if.mask = extractvalue { i1, i1 } %if, 1
  %if.bool.ext = zext i1 %if.bool to i32
  store volatile i32 %if.bool.ext, ptr addrspace(1) undef
  store volatile i1 %if.mask, ptr addrspace(1) undef
  ret void
}

declare { i1, i1 } @llvm.amdgcn.if(i1) #0
declare { i1, i1 } @llvm.amdgcn.else(i1) #0
declare i1 @llvm.amdgcn.if.break(i1, i1) #1
declare i1 @llvm.amdgcn.loop(i1) #1

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nounwind readnone }
