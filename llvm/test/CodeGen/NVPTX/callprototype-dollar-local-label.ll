; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_75 | FileCheck %s
; RUN: %if ptxas-sm_75 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_75 | %ptxas-verify --compile-only -arch=sm_75 %}

target triple = "nvptx64-nvidia-cuda"

@"$L__prototype_0" = addrspace(1) global ptr null, align 8
@"$L__prototype_00" = addrspace(1) global i32 7, align 4

define i32 @call_via_dollar_prototype_0(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK-DAG: .visible .global .align 8 .u64 $L__prototype_0;
; CHECK-DAG: .visible .global .align 4 .u32 $L__prototype_00 = 7;
; CHECK-LABEL: call_via_dollar_prototype_0(
; CHECK: $L__prototype_01 : .callprototype (.param .b32 _) _ (.param .b32 _, .param .b32 _, .param .b32 _, .param .b32 _);
; CHECK: ld.global.{{u|b}}64 {{%rd[0-9]+}}, [$L__prototype_0];
; CHECK: call (retval0), %rd{{[0-9]+}}, (param0, param1, param2, param3), $L__prototype_01;
  %fp = load ptr, ptr addrspace(1) @"$L__prototype_0", align 8
  %ret = call i32 %fp(i32 %a, i32 %b, i32 %c, i32 %d)
  ret i32 %ret
}
