; Multi-register aggregate arms (llvm.sincos, llvm.frexp) of a select or PHI
; are rebuilt with insertvalue, each chain from its own typed placeholder, and
; a PHI arm once per incoming block.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,NOEXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - | FileCheck %s --check-prefixes=CHECK,EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_poison_freeze %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Pair:]] = OpTypeStruct %[[#Float]] %[[#Float]]
; CHECK-DAG: %[[#Frexp:]] = OpTypeStruct %[[#Float]] %[[#Int]]
; NOEXT-DAG: %[[#PairInit:]] = OpUndef %[[#Pair]]
; NOEXT-DAG: %[[#FrexpInit:]] = OpUndef %[[#Frexp]]
; EXT-DAG: %[[#PairInit:]] = OpPoisonKHR %[[#Pair]]
; EXT-DAG: %[[#FrexpInit:]] = OpPoisonKHR %[[#Frexp]]
; CHECK: OpCompositeInsert %[[#Pair]] %[[#]] %[[#PairInit]] 0
; CHECK: OpSelect %[[#Pair]]
; CHECK: OpCompositeInsert %[[#Frexp]] %[[#]] %[[#FrexpInit]] 0
; CHECK: OpSelect %[[#Frexp]]
; CHECK: OpFunction %[[#Pair]]
; CHECK: OpCompositeInsert %[[#Pair]] %[[#]] %[[#]] 0
; CHECK: %[[#Arm:]] = OpCompositeInsert %[[#Pair]] %[[#]] %[[#]] 1
; CHECK-NOT: OpCompositeInsert
; CHECK: OpPhi %[[#Pair]] %[[#Arm]] %[[#]] %[[#]] %[[#]]{{$}}

define spir_kernel void @k(ptr addrspace(1) %out, float %a, i1 %c) {
  %sc = call { float, float } @llvm.sincos.f32(float %a)
  %s1 = select i1 %c, { float, float } zeroinitializer, { float, float } %sc
  %x = extractvalue { float, float } %s1, 0
  store float %x, ptr addrspace(1) %out
  %fe = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  %s2 = select i1 %c, { float, i32 } zeroinitializer, { float, i32 } %fe
  %e = extractvalue { float, i32 } %s2, 1
  %p = getelementptr inbounds i8, ptr addrspace(1) %out, i64 4
  store i32 %e, ptr addrspace(1) %p
  ret void
}

define { float, float } @phi(i32 %x, float %a) {
entry:
  %sc = call { float, float } @llvm.sincos.f32(float %a)
  switch i32 %x, label %join [
    i32 1, label %join
    i32 2, label %other
  ]
other:
  br label %join
join:
  %p = phi { float, float } [ %sc, %entry ], [ %sc, %entry ], [ zeroinitializer, %other ]
  ret { float, float } %p
}
