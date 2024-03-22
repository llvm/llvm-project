; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefix=CHECK %s

declare void @function1()

declare void @function2() "amdgpu-no-hostcall-ptr"

; Function Attrs: noinline
define void @function3(ptr addrspace(4) %argptr, ptr addrspace(1) %sink) noinline {
  store ptr addrspace(4) %argptr, ptr addrspace(1) %sink, align 8
  ret void
}

; Function Attrs: noinline
define void @function4(i64 %arg, ptr %a) noinline {
  store i64 %arg, ptr %a
  ret void
}

; Function Attrs: noinline
define void @function5(ptr addrspace(4) %ptr, ptr %sink) noinline {
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 8
  %x = load i64, ptr addrspace(4) %gep
  store i64 %x, ptr %sink
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() nounwind readnone speculatable willreturn

; CHECK: amdhsa.kernels:
; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel10
define amdgpu_kernel void @test_kernel10(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  store i8 3, ptr %a, align 1
  ret void
}

; Call to an extern function

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel20
define amdgpu_kernel void @test_kernel20(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  call void @function1()
  store i8 3, ptr %a, align 1
  ret void
}

; Explicit attribute on kernel

; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel21
define amdgpu_kernel void @test_kernel21(ptr %a) "amdgpu-implicitarg-num-bytes"="48" "amdgpu-no-hostcall-ptr" {
  call void @function1()
  store i8 3, ptr %a, align 1
  ret void
}

; Explicit attribute on extern callee

; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel22
define amdgpu_kernel void @test_kernel22(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  call void @function2()
  store i8 3, ptr %a, align 1
  ret void
}

; Access more bytes than the pointer size

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel30
define amdgpu_kernel void @test_kernel30(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 16
  %x = load i128, ptr addrspace(4) %gep
  store i128 %x, ptr %a
  ret void
}

; Typical load of hostcall buffer pointer

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel40
define amdgpu_kernel void @test_kernel40(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 24
  %x = load i64, ptr addrspace(4) %gep
  store i64 %x, ptr %a
  ret void
}

; Typical usage, overriden by explicit attribute on kernel

; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel41
define amdgpu_kernel void @test_kernel41(ptr %a) "amdgpu-implicitarg-num-bytes"="48" "amdgpu-no-hostcall-ptr" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 24
  %x = load i64, ptr addrspace(4) %gep
  store i64 %x, ptr %a
  ret void
}

; Access to implicit arg before the hostcall pointer

; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel42
define amdgpu_kernel void @test_kernel42(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 16
  %x = load i64, ptr addrspace(4) %gep
  store i64 %x, ptr %a
  ret void
}

; Access to implicit arg after the hostcall pointer

; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel43
define amdgpu_kernel void @test_kernel43(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 32
  %x = load i64, ptr addrspace(4) %gep
  store i64 %x, ptr %a
  ret void
}

; Accessing a byte just before the hostcall pointer

; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel44
define amdgpu_kernel void @test_kernel44(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 23
  %x = load i8, ptr addrspace(4) %gep, align 1
  store i8 %x, ptr %a, align 1
  ret void
}

; Accessing a byte inside the hostcall pointer

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel45
define amdgpu_kernel void @test_kernel45(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 24
  %x = load i8, ptr addrspace(4) %gep, align 1
  store i8 %x, ptr %a, align 1
  ret void
}

; Accessing a byte inside the hostcall pointer

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel46
define amdgpu_kernel void @test_kernel46(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 31
  %x = load i8, ptr addrspace(4) %gep, align 1
  store i8 %x, ptr %a, align 1
  ret void
}

; Accessing a byte just after the hostcall pointer

; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel47
define amdgpu_kernel void @test_kernel47(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 32
  %x = load i8, ptr addrspace(4) %gep, align 1
  store i8 %x, ptr %a, align 1
  ret void
}

; Access with an unknown offset

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel50
define amdgpu_kernel void @test_kernel50(ptr %a, i32 %b) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i32 %b
  %x = load i8, ptr addrspace(4) %gep, align 1
  store i8 %x, ptr %a, align 1
  ret void
}

; Multiple geps reaching the hostcall pointer argument.

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel51
define amdgpu_kernel void @test_kernel51(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep1 = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 16
  %gep2 = getelementptr inbounds i8, ptr addrspace(4) %gep1, i64 8
  %x = load i8, ptr addrspace(4) %gep2, align 1
  store i8 %x, ptr %a, align 1
  ret void
}

; Multiple geps not reaching the hostcall pointer argument.

; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel52
define amdgpu_kernel void @test_kernel52(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep1 = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 16
  %gep2 = getelementptr inbounds i8, ptr addrspace(4) %gep1, i64 16
  %x = load i8, ptr addrspace(4) %gep2, align 1
  store i8 %x, ptr %a, align 1
  ret void
}

; Hostcall pointer used inside a function call

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel60
define amdgpu_kernel void @test_kernel60(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 24
  %x = load i64, ptr addrspace(4) %gep
  call void @function4(i64 %x, ptr %a)
  ret void
}

; Hostcall pointer retrieved inside a function call; chain of geps

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel61
define amdgpu_kernel void @test_kernel61(ptr %a) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 16
  call void @function5(ptr addrspace(4) %gep, ptr %a)
  ret void
}

; Pointer captured

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel70
define amdgpu_kernel void @test_kernel70(ptr addrspace(1) %sink) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i32 42
  store ptr addrspace(4) %gep, ptr addrspace(1) %sink, align 8
  ret void
}

; Pointer captured inside function call

; CHECK:  - .args:
; CHECK: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel71
define amdgpu_kernel void @test_kernel71(ptr addrspace(1) %sink) "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i32 42
  call void @function3(ptr addrspace(4) %gep, ptr addrspace(1) %sink)
  ret void
}

; Ineffective pointer capture

; CHECK:  - .args:
; CHECK-NOT: hidden_hostcall_buffer
; CHECK-LABEL:    .name:           test_kernel72
define amdgpu_kernel void @test_kernel72() "amdgpu-implicitarg-num-bytes"="48" {
  %ptr = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep = getelementptr inbounds i8, ptr addrspace(4) %ptr, i32 42
  store ptr addrspace(4) %gep, ptr addrspace(1) undef, align 8
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
