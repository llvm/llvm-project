; RUN: opt -O3 -S < %s | FileCheck %s
; Show 'optnone' suppresses optimizations.

; Two attribute groups that differ only by 'optnone'.
; 'optnone' requires 'noinline' so #0 is 'noinline' by itself,
; even though it would otherwise be irrelevant to this example.
attributes #0 = { noinline }
attributes #1 = { noinline optnone }

; int iadd(int a, int b){ return a + b; }

define i32 @iadd_optimize(i32 %a, i32 %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

; CHECK-LABEL: @iadd_optimize
; CHECK-NOT: alloca
; CHECK-NOT: store
; CHECK-NOT: load
; CHECK: ret

define i32 @iadd_optnone(i32 %a, i32 %b) #1 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

; CHECK-LABEL: @iadd_optnone
; CHECK: alloca i32
; CHECK: alloca i32
; CHECK: store i32
; CHECK: store i32
; CHECK: load i32
; CHECK: load i32
; CHECK: add nsw i32
; CHECK: ret i32

; float fsub(float a, float b){ return a - b; }

define float @fsub_optimize(float %a, float %b) #0 {
entry:
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  store float %b, ptr %b.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = load float, ptr %b.addr, align 4
  %sub = fsub float %0, %1
  ret float %sub
}

; CHECK-LABEL: @fsub_optimize
; CHECK-NOT: alloca
; CHECK-NOT: store
; CHECK-NOT: load
; CHECK: ret

define float @fsub_optnone(float %a, float %b) #1 {
entry:
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  store float %b, ptr %b.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = load float, ptr %b.addr, align 4
  %sub = fsub float %0, %1
  ret float %sub
}

; CHECK-LABEL: @fsub_optnone
; CHECK: alloca float
; CHECK: alloca float
; CHECK: store float
; CHECK: store float
; CHECK: load float
; CHECK: load float
; CHECK: fsub float
; CHECK: ret float

; typedef float __attribute__((ext_vector_type(4))) float4;
; float4 vmul(float4 a, float4 b){ return a * b; }

define <4 x float> @vmul_optimize(<4 x float> %a, <4 x float> %b) #0 {
entry:
  %a.addr = alloca <4 x float>, align 16
  %b.addr = alloca <4 x float>, align 16
  store <4 x float> %a, ptr %a.addr, align 16
  store <4 x float> %b, ptr %b.addr, align 16
  %0 = load <4 x float>, ptr %a.addr, align 16
  %1 = load <4 x float>, ptr %b.addr, align 16
  %mul = fmul <4 x float> %0, %1
  ret <4 x float> %mul
}

; CHECK-LABEL: @vmul_optimize
; CHECK-NOT: alloca
; CHECK-NOT: store
; CHECK-NOT: load
; CHECK: ret

define <4 x float> @vmul_optnone(<4 x float> %a, <4 x float> %b) #1 {
entry:
  %a.addr = alloca <4 x float>, align 16
  %b.addr = alloca <4 x float>, align 16
  store <4 x float> %a, ptr %a.addr, align 16
  store <4 x float> %b, ptr %b.addr, align 16
  %0 = load <4 x float>, ptr %a.addr, align 16
  %1 = load <4 x float>, ptr %b.addr, align 16
  %mul = fmul <4 x float> %0, %1
  ret <4 x float> %mul
}

; CHECK-LABEL: @vmul_optnone
; CHECK: alloca <4 x float>
; CHECK: alloca <4 x float>
; CHECK: store <4 x float>
; CHECK: store <4 x float>
; CHECK: load <4 x float>
; CHECK: load <4 x float>
; CHECK: fmul <4 x float>
; CHECK: ret
