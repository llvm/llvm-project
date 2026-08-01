; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix CHECK-DEFAULT
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix CHECK-DEFAULT-32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -nvptx-short-ptr | FileCheck %s --check-prefixes CHECK-SHORT-SHARED,CHECK-SHORT-CONST,CHECK-SHORT-LOCAL

; RUN: %if ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -nvptx-short-ptr | %ptxas-verify %}

; CHECK-DEFAULT: .visible .shared .align 8 .u64 s
; CHECK-DEFAULT-32: .visible .shared .align 8 .u32 s
; CHECK-SHORT-SHARED: .visible .shared .align 8 .u32 s
@s = local_unnamed_addr addrspace(3) global ptr addrspace(3) null, align 8

; CHECK-DEFAULT: .visible .const .align 8 .u64 c
; CHECK-DEFAULT-32: .visible .const .align 8 .u32 c
; CHECK-SHORT-CONST: .visible .const .align 8 .u32 c
@c = local_unnamed_addr addrspace(4) global ptr addrspace(4) null, align 8

declare void @use(i8 %arg);

; CHECK-DEFAULT: .param .b64 test1_param_0
; CHECK-DEFAULT-32: .param .b32 test1_param_0
; CHECK-SHORT-LOCAL: .param .b32 test1_param_0
define void @test1(ptr addrspace(5) %local) {
  ; CHECK-DEFAULT: ld.param.b64 %rd{{.*}}, [test1_param_0];
  ; CHECK-DEFAULT-32:  ld.param.b32 %r{{.*}}, [test1_param_0];
  ; CHECK-SHORT-LOCAL: ld.param.b32 %r{{.*}}, [test1_param_0];
  %v = load i8, ptr addrspace(5) %local
  call void @use(i8 %v)
  ret void
}

define void @test2() {
  %v = alloca i8
  %cast = addrspacecast ptr %v to ptr addrspace(5)
  ; CHECK-DEFAULT: .param .b64 param0;
  ; CHECK-DEFAULT: add.u64 %rd{{.*}}, %SPL, 0;
  ; CHECK-DEFAULT: st.param.b64
  ; CHECK-DEFAULT-32: .param .b32 param0;
  ; CHECK-DEFAULT-32: add.u32 %r{{.*}}, %SPL, 0;
  ; CHECK-DEFAULT-32: st.param.b32
  ; CHECK-SHORT-LOCAL: .reg .b32 %SPL;
  ; CHECK-SHORT-LOCAL: .param .b32 param0;
  ; CHECK-SHORT-LOCAL: add.u32 %r{{.*}}, %SPL, 0;
  ; CHECK-SHORT-LOCAL: st.param.b32
  call void @test1(ptr addrspace(5) %cast)
  ret void
}

; Exercise the mixed-width prologue for a generic stack temporary.
define <4 x float> @test3(<4 x float> %v, i32 %i, float %x) {
  ; CHECK-DEFAULT: mov.b64 %SPL, __local_depot2;
  ; CHECK-DEFAULT-NEXT: cvta.local.u64 %SP, %SPL;
  ; CHECK-DEFAULT-32: mov.b32 %SPL, __local_depot2;
  ; CHECK-DEFAULT-32-NEXT: cvta.local.u32 %SP, %SPL;
  ; CHECK-SHORT-LOCAL: mov.b32 %SPL, __local_depot2;
  ; CHECK-SHORT-LOCAL-NEXT: cvt.u64.u32 %SP, %SPL;
  ; CHECK-SHORT-LOCAL-NEXT: cvta.local.u64 %SP, %SP;
  %r = insertelement <4 x float> %v, float %x, i32 %i
  ret <4 x float> %r
}
