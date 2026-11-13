; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx \
; RUN:   -stop-after=prolog-epilog -verify-machineinstrs < %s | FileCheck %s

; Check that local HLX objects use the VTCM stack.

; CHECK-LABEL: name: local_vectors
; CHECK: V2_vallocframe_imm 2
; CHECK: $[[VSP:r[0-9]+]] = A2_tfrcrr $vsp
; CHECK: A2_addi {{.*}}$[[VSP]], 2048
; CHECK: V2_vdeallocframe_imm 2

define void @local_vectors(ptr %in, ptr %out) {
entry:
  %v0 = alloca <512 x i32>, align 2048
  %v1 = alloca <512 x i32>, align 2048
  %x = load <512 x i32>, ptr %in, align 2048
  store <512 x i32> %x, ptr %v0, align 2048
  store <512 x i32> %x, ptr %v1, align 2048
  %y = load <512 x i32>, ptr %v0, align 2048
  store <512 x i32> %y, ptr %out, align 2048
  ret void
}

%hlx.array.element = type { <1024 x i16> }

; Check recursive classification of arrays and structures containing HLX
; vectors.

; CHECK-LABEL: name: vector_array
; CHECK: V2_vallocframe_imm 10
; CHECK-DAG: A2_addi {{.*}}, 4096
; CHECK-DAG: A2_addi {{.*}}, 12288
; CHECK: V2_vdeallocframe_imm 10

define void @vector_array(ptr %in) {
entry:
  %array = alloca [10 x %hlx.array.element], align 2048
  %elt2 = getelementptr [10 x %hlx.array.element], ptr %array, i32 0, i32 2
  %elt6 = getelementptr [10 x %hlx.array.element], ptr %array, i32 0, i32 6
  %x = load <1024 x i16>, ptr %in, align 2048
  store <1024 x i16> %x, ptr %elt2, align 2048
  store <1024 x i16> %x, ptr %elt6, align 2048
  ret void
}

; A scalar object is not a VTCM object merely because its size and alignment
; match an HLX vector.

; CHECK-LABEL: name: aligned_scalar
; CHECK-NOT: V2_vallocframe
; CHECK-NOT: A2_tfrcrr $vsp
; CHECK-NOT: V2_vdeallocframe

define i64 @aligned_scalar(i64 %x) {
entry:
  %object = alloca [256 x i64], align 2048
  %slot = getelementptr [256 x i64], ptr %object, i32 0, i32 255
  store volatile i64 %x, ptr %slot, align 8
  %result = load volatile i64, ptr %slot, align 8
  ret i64 %result
}
