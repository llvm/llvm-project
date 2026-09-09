; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Floating-point globals are declared with a bit type and initialized with
; their bit pattern. PTX has no 16-bit floating-point literal, so this is the
; only form available for half and bfloat, and the .f32 and .f64 literals are
; themselves just bit patterns in hex.

; CHECK-DAG: .visible .global .align 2 .b16 h = 0x3C00;
@h = addrspace(1) global half 0xH3C00

; CHECK-DAG: .visible .global .align 2 .b16 bf = 0x3F80;
@bf = addrspace(1) global bfloat 0xR3F80

; CHECK-DAG: .visible .global .align 4 .b32 f = 0x3F800000;
@f = addrspace(1) global float 1.0

; CHECK-DAG: .visible .global .align 8 .b64 d = 0x4000000000000000;
@d = addrspace(1) global double 2.0

; The bit pattern is padded out to the width of the type.
; CHECK-DAG: .visible .global .align 2 .b16 h_denormal = 0x0001;
@h_denormal = addrspace(1) global half 0xH0001

; Signed zeros and NaNs round-trip as the bits they are.
; CHECK-DAG: .visible .global .align 4 .b32 f_negzero = 0x80000000;
@f_negzero = addrspace(1) global float -0.0

; CHECK-DAG: .visible .global .align 8 .b64 d_nan = 0x7FF8000000000000;
@d_nan = addrspace(1) global double 0x7FF8000000000000

; A zero initializer is treated as no value specified.
; CHECK-DAG: .visible .global .align 2 .b16 h_zero;
@h_zero = addrspace(1) global half 0xH0000

; A declaration has no initializer, but must still agree on the type.
; CHECK-DAG: .extern .global .align 4 .b32 f_decl;
@f_decl = external addrspace(1) global float

; CHECK-DAG: .visible .const .align 4 .b32 f_const = 0x40600000;
@f_const = addrspace(4) global float 3.5

; Aggregates keep going through the byte-buffered path.
; CHECK-DAG: .visible .global .align 2 .b8 h_arr[4] = {0, 60, 0, 64};
@h_arr = addrspace(1) global [2 x half] [half 0xH3C00, half 0xH4000]

define ptx_kernel void @use(ptr %p) {
  %v = load float, ptr addrspace(1) @f_decl
  ; Instruction immediates still need a floating-point literal; ptxas rejects
  ; an integer where an .f32 immediate is expected.
  ; CHECK: add.rn.f32 %r{{[0-9]+}}, %r{{[0-9]+}}, 0f3FC00000;
  %a = fadd float %v, 1.5
  store float %a, ptr %p
  ret void
}
