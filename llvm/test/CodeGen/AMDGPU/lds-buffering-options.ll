; NOTE: Do not autogenerate. This tests experimental candidate controls.
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='amdgpu-lds-buffering<max-bytes=64>' -S < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='amdgpu-lds-buffering<max-bytes=64;min-align=4>' -S < %s | FileCheck %s --check-prefix=RELAXED
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='amdgpu-lds-buffering<max-bytes=64;min-align=4;only-candidate=1>' -S < %s | FileCheck %s --check-prefix=SECOND
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='amdgpu-lds-buffering<max-bytes=64;min-align=4;only-candidate=1;mode=shadow-lds>' -S < %s | FileCheck %s --check-prefix=SHADOW
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='amdgpu-lds-buffering<max-bytes=64;min-align=4;only-candidate=1;mode=irrelevant-lds>' -S < %s | FileCheck %s --check-prefix=CONTROL

; DEFAULT-LABEL: define amdgpu_kernel void @natural_alignment(
; DEFAULT: %first = load i32, ptr addrspace(1) %p, align 4
; DEFAULT: store i32 %first, ptr addrspace(1) %p, align 4
; DEFAULT: %second = load i32, ptr addrspace(1) %q, align 4
; DEFAULT: store i32 %second, ptr addrspace(1) %q, align 4

; RELAXED: @natural_alignment.ldsbuf = internal unnamed_addr addrspace(3) global
; RELAXED: @natural_alignment.ldsbuf.1 = internal unnamed_addr addrspace(3) global
; RELAXED-LABEL: define amdgpu_kernel void @natural_alignment(
; RELAXED-NOT: load i32, ptr addrspace(1) %p
; RELAXED: call void @llvm.memcpy.p3.p1.i64({{.*}}%p, i64 4, i1 false)
; RELAXED: call void @llvm.memcpy.p1.p3.i64({{.*}}%p, {{.*}}i64 4, i1 false)
; RELAXED: call void @llvm.memcpy.p3.p1.i64({{.*}}%q, i64 4, i1 false)
; RELAXED: call void @llvm.memcpy.p1.p3.i64({{.*}}%q, {{.*}}i64 4, i1 false)

; SECOND: @natural_alignment.ldsbuf = internal unnamed_addr addrspace(3) global
; SECOND-NOT: @natural_alignment.ldsbuf.1 =
; SECOND-LABEL: define amdgpu_kernel void @natural_alignment(
; SECOND: %first = load i32, ptr addrspace(1) %p, align 4
; SECOND: store i32 %first, ptr addrspace(1) %p, align 4
; SECOND-NOT: load i32, ptr addrspace(1) %q
; SECOND: call void @llvm.memcpy.p3.p1.i64({{.*}}%q, i64 4, i1 false)
; SECOND: call void @llvm.memcpy.p1.p3.i64({{.*}}%q, {{.*}}i64 4, i1 false)

; SHADOW: @natural_alignment.ldsbuf = internal unnamed_addr addrspace(3) global
; SHADOW-LABEL: define amdgpu_kernel void @natural_alignment(
; SHADOW: %second = load i32, ptr addrspace(1) %q, align 4
; SHADOW: store volatile i32 %second, ptr addrspace(3) {{.*}}, align 4
; SHADOW: store i32 %second, ptr addrspace(1) %q, align 4
; SHADOW: %ldsbuf.control = load volatile i32, ptr addrspace(3) {{.*}}, align 4
; SHADOW-NOT: call void @llvm.memcpy

; CONTROL: @natural_alignment.ldsbuf = internal unnamed_addr addrspace(3) global
; CONTROL-NOT: @natural_alignment.ldsbuf.1 =
; CONTROL-LABEL: define amdgpu_kernel void @natural_alignment(
; CONTROL: %first = load i32, ptr addrspace(1) %p, align 4
; CONTROL: store i32 %first, ptr addrspace(1) %p, align 4
; CONTROL: %second = load i32, ptr addrspace(1) %q, align 4
; CONTROL: store volatile i32 0, ptr addrspace(3) {{.*}}, align 4
; CONTROL: store i32 %second, ptr addrspace(1) %q, align 4
; CONTROL: %ldsbuf.control = load volatile i32, ptr addrspace(3) {{.*}}, align 4
; CONTROL-NOT: call void @llvm.memcpy

define amdgpu_kernel void @natural_alignment(ptr addrspace(1) %p,
                                              ptr addrspace(1) %q,
                                              ptr addrspace(1) %out) #0 {
entry:
  %first = load i32, ptr addrspace(1) %p, align 4
  store i32 1, ptr addrspace(1) %out, align 4
  store i32 %first, ptr addrspace(1) %p, align 4
  %second = load i32, ptr addrspace(1) %q, align 4
  store i32 2, ptr addrspace(1) %out, align 4
  store i32 %second, ptr addrspace(1) %q, align 4
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "uniform-work-group-size"="true" }
