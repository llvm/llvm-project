; RUN: llc --filetype=obj %s --stop-after=dxil-write-bitcode -o %t.bc
; RUN: dxil-dis %t.bc -o - | FileCheck %s

; Verify that the DXIL reader supports the atomicrmw and the cmpxchg records.
; A float allocation needs a pointer cast, because the atomic value is an
; integer and a DXIL pointer carries the element type.

target triple = "dxil-pc-shadermodel6.0-compute"

@gsm = internal addrspace(3) global i32 zeroinitializer, align 4
@gsm_arr = internal addrspace(3) global [4 x i32] zeroinitializer, align 4
@gsm_float = internal addrspace(3) global float zeroinitializer, align 4
@gsm_farr = internal addrspace(3) global [4 x float] zeroinitializer, align 4

; CHECK-LABEL: define void @main()
; An integer allocation already agrees with the value, so it gets no cast.
; CHECK-NEXT: atomicrmw add i32 addrspace(3)* @gsm, i32 1 monotonic
; CHECK-NEXT: [[INDEX:%.*]] = load i32, i32 addrspace(3)* @gsm, align 4
; CHECK-NEXT: [[INT_MUL:%.*]] = mul i32 [[INDEX]], 1
; CHECK-NEXT: [[INT_INDEX:%.*]] = add i32 0, [[INT_MUL]]
; CHECK-NEXT: [[INT_GEP:%.*]] = getelementptr [4 x i32], [4 x i32] addrspace(3)* @gsm_arr, i32 0, i32 [[INT_INDEX]]
; CHECK-NEXT: atomicrmw add i32 addrspace(3)* [[INT_GEP]], i32 6 monotonic
; CHECK-NEXT: [[P0:%.*]] = bitcast float addrspace(3)* @gsm_float to i32 addrspace(3)*
; CHECK-NEXT: atomicrmw xchg i32 addrspace(3)* [[P0]], i32 2 monotonic
; CHECK-NEXT: [[FLOAT_MUL:%.*]] = mul i32 [[INDEX]], 1
; CHECK-NEXT: [[FLOAT_INDEX:%.*]] = add i32 0, [[FLOAT_MUL]]
; CHECK-NEXT: [[FLOAT_GEP:%.*]] = getelementptr [4 x float], [4 x float] addrspace(3)* @gsm_farr, i32 0, i32 [[FLOAT_INDEX]]
; CHECK-NEXT: [[P1:%.*]] = bitcast float addrspace(3)* [[FLOAT_GEP]] to i32 addrspace(3)*
; CHECK-NEXT: atomicrmw xchg i32 addrspace(3)* [[P1]], i32 3 monotonic
; CHECK-NEXT: [[P2:%.*]] = bitcast float addrspace(3)* @gsm_float to i32 addrspace(3)*
; CHECK-NEXT: cmpxchg i32 addrspace(3)* [[P2]], i32 4, i32 5 monotonic monotonic

define void @main() #0 {
  %old = atomicrmw add ptr addrspace(3) @gsm, i32 1 monotonic
  %index = load i32, ptr addrspace(3) @gsm, align 4
  %int_gep = getelementptr [4 x i32], ptr addrspace(3) @gsm_arr, i32 0, i32 %index
  %arr = atomicrmw add ptr addrspace(3) %int_gep, i32 6 monotonic
  %f = atomicrmw xchg ptr addrspace(3) @gsm_float, i32 2 monotonic
  %gep = getelementptr [4 x float], ptr addrspace(3) @gsm_farr, i32 0, i32 %index
  %g = atomicrmw xchg ptr addrspace(3) %gep, i32 3 monotonic
  %c = cmpxchg ptr addrspace(3) @gsm_float, i32 4, i32 5 monotonic monotonic
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
