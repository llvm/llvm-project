; RUN: opt -mtriple amdgcn-mesa-mesa3d -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; Tests control flow intrinsics that should be treated as uniform

; CHECK: for function 'test_if_break':
; CHECK: DIVERGENT: %cond = icmp eq i32 %arg0, 0
; CHECK-NOT: DIVERGENT
; CHECK: ret void
define amdgpu_ps void @test_if_break(i32 %arg0, i64 inreg %saved) {
entry:
  %cond = icmp eq i32 %arg0, 0
  %break = call i64 @llvm.amdgcn.if.break.i64.i64(i1 %cond, i64 %saved)
  store volatile i64 %break, ptr addrspace(1) undef
  ret void
}

; CHECK: for function 'test_if':
; CHECK: DIVERGENT: %cond = icmp eq i32 %arg0, 0
; CHECK-NEXT: DIVERGENT: %if = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %cond)
; CHECK-NEXT: DIVERGENT: %if.bool = extractvalue { i1, i64 } %if, 0
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: %if.bool.ext = zext i1 %if.bool to i32
define void @test_if(i32 %arg0) {
entry:
  %cond = icmp eq i32 %arg0, 0
  %if = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %cond)
  %if.bool = extractvalue { i1, i64 } %if, 0
  %if.mask = extractvalue { i1, i64 } %if, 1
  %if.bool.ext = zext i1 %if.bool to i32
  store volatile i32 %if.bool.ext, ptr addrspace(1) undef
  store volatile i64 %if.mask, ptr addrspace(1) undef
  ret void
}

; The result should still be treated as divergent, even with a uniform source.
; CHECK: for function 'test_if_uniform':
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: %if = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %cond)
; CHECK-NEXT: DIVERGENT: %if.bool = extractvalue { i1, i64 } %if, 0
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: %if.bool.ext = zext i1 %if.bool to i32
define amdgpu_ps void @test_if_uniform(i32 inreg %arg0) {
entry:
  %cond = icmp eq i32 %arg0, 0
  %if = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %cond)
  %if.bool = extractvalue { i1, i64 } %if, 0
  %if.mask = extractvalue { i1, i64 } %if, 1
  %if.bool.ext = zext i1 %if.bool to i32
  store volatile i32 %if.bool.ext, ptr addrspace(1) undef
  store volatile i64 %if.mask, ptr addrspace(1) undef
  ret void
}

; CHECK: for function 'test_loop_uniform':
; CHECK: DIVERGENT: %loop = call i1 @llvm.amdgcn.loop.i64(i64 %mask)
define amdgpu_ps void @test_loop_uniform(i64 inreg %mask) {
entry:
  %loop = call i1 @llvm.amdgcn.loop.i64(i64 %mask)
  %loop.ext = zext i1 %loop to i32
  store volatile i32 %loop.ext, ptr addrspace(1) undef
  ret void
}

; CHECK: for function 'test_else':
; CHECK: DIVERGENT: %else = call { i1, i64 } @llvm.amdgcn.else.i64.i64(i64 %mask)
; CHECK: DIVERGENT:       %else.bool = extractvalue { i1, i64 } %else, 0
; CHECK: {{^[ \t]+}}%else.mask = extractvalue { i1, i64 } %else, 1
define amdgpu_ps void @test_else(i64 inreg %mask) {
entry:
  %else = call { i1, i64 } @llvm.amdgcn.else.i64.i64(i64 %mask)
  %else.bool = extractvalue { i1, i64 } %else, 0
  %else.mask = extractvalue { i1, i64 } %else, 1
  %else.bool.ext = zext i1 %else.bool to i32
  store volatile i32 %else.bool.ext, ptr addrspace(1) undef
  store volatile i64 %else.mask, ptr addrspace(1) undef
  ret void
}

; This case is probably always broken
; CHECK: for function 'test_else_divergent_mask':
; CHECK: DIVERGENT: %if = call { i1, i64 } @llvm.amdgcn.else.i64.i64(i64 %mask)
; CHECK-NEXT: DIVERGENT: %if.bool = extractvalue { i1, i64 } %if, 0
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT: %if.bool.ext = zext i1 %if.bool to i32
define void @test_else_divergent_mask(i64 %mask) {
entry:
  %if = call { i1, i64 } @llvm.amdgcn.else.i64.i64(i64 %mask)
  %if.bool = extractvalue { i1, i64 } %if, 0
  %if.mask = extractvalue { i1, i64 } %if, 1
  %if.bool.ext = zext i1 %if.bool to i32
  store volatile i32 %if.bool.ext, ptr addrspace(1) undef
  store volatile i64 %if.mask, ptr addrspace(1) undef
  ret void
}

declare { i1, i64 } @llvm.amdgcn.if.i64(i1) #0
declare { i1, i64 } @llvm.amdgcn.else.i64.i64(i64) #0
declare i64 @llvm.amdgcn.if.break.i64.i64(i1, i64) #1
declare i1 @llvm.amdgcn.loop.i64(i64) #1

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nounwind readnone }
