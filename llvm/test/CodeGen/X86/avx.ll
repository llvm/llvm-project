; RUN: llc < %s -mtriple=i686-apple-darwin -mcpu=corei7-avx | FileCheck %s --check-prefixes=X32,CHECK
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s

define <4 x i32> @blendvb_fallback_v4i32(<4 x i1> %mask, <4 x i32> %x, <4 x i32> %y) {
; CHECK-LABEL: @blendvb_fallback_v4i32
; CHECK: vblendvps
; CHECK: ret
  %ret = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %ret
}

define <8 x i32> @blendvb_fallback_v8i32(<8 x i1> %mask, <8 x i32> %x, <8 x i32> %y) {
; CHECK-LABEL: @blendvb_fallback_v8i32
; CHECK: vblendvps
; CHECK: ret
  %ret = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %y
  ret <8 x i32> %ret
}

define <8 x float> @blendvb_fallback_v8f32(<8 x i1> %mask, <8 x float> %x, <8 x float> %y) {
; CHECK-LABEL: @blendvb_fallback_v8f32
; CHECK: vblendvps
; CHECK: ret
  %ret = select <8 x i1> %mask, <8 x float> %x, <8 x float> %y
  ret <8 x float> %ret
}

declare <4 x float> @llvm.x86.sse41.insertps(<4 x float>, <4 x float>, i32) nounwind readnone

define <4 x float> @insertps_from_vector_load(<4 x float> %a, <4 x float>* nocapture readonly %pb) {
; CHECK-LABEL: insertps_from_vector_load:
; On X32, account for the argument's move to registers
; X32: movl    4(%esp), %eax
; CHECK-NOT: mov
; CHECK: vinsertps    $48, (%{{...}}), {{.*#+}} xmm0 = xmm0[0,1,2],mem[0]
; CHECK-NEXT: ret
  %1 = load <4 x float>, <4 x float>* %pb, align 16
  %2 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %1, i32 48)
  ret <4 x float> %2
}

;; Use a non-zero CountS for insertps
define <4 x float> @insertps_from_vector_load_offset(<4 x float> %a, <4 x float>* nocapture readonly %pb) {
; CHECK-LABEL: insertps_from_vector_load_offset:
; On X32, account for the argument's move to registers
; X32: movl    4(%esp), %eax
; CHECK-NOT: mov
;; Try to match a bit more of the instr, since we need the load's offset.
; CHECK: vinsertps    $32, 4(%{{...}}), {{.*#+}} xmm0 = xmm0[0,1],mem[0],xmm0[3]
; CHECK-NEXT: ret
  %1 = load <4 x float>, <4 x float>* %pb, align 16
  %2 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %1, i32 96)
  ret <4 x float> %2
}

define <4 x float> @insertps_from_vector_load_offset_2(<4 x float> %a, <4 x float>* nocapture readonly %pb, i64 %index) {
; CHECK-LABEL: insertps_from_vector_load_offset_2:
; On X32, account for the argument's move to registers
; X32: movl    4(%esp), %eax
; X32: movl    8(%esp), %ecx
; CHECK-NOT: mov
;; Try to match a bit more of the instr, since we need the load's offset.
; CHECK: vinsertps    $0, 12(%{{...}},%{{...}}), {{.*#+}} xmm0 = mem[0],xmm0[1,2,3]
; CHECK-NEXT: ret
  %1 = getelementptr inbounds <4 x float>, <4 x float>* %pb, i64 %index
  %2 = load <4 x float>, <4 x float>* %1, align 16
  %3 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %2, i32 192)
  ret <4 x float> %3
}

define <4 x float> @insertps_from_broadcast_loadf32(<4 x float> %a, float* nocapture readonly %fb, i64 %index) {
; CHECK-LABEL: insertps_from_broadcast_loadf32:
; On X32, account for the arguments' move to registers
; X32: movl    8(%esp), %eax
; X32: movl    4(%esp), %ecx
; CHECK-NOT: mov
; CHECK: insertps    $48
; CHECK-NEXT: ret
  %1 = getelementptr inbounds float, float* %fb, i64 %index
  %2 = load float, float* %1, align 4
  %3 = insertelement <4 x float> undef, float %2, i32 0
  %4 = insertelement <4 x float> %3, float %2, i32 1
  %5 = insertelement <4 x float> %4, float %2, i32 2
  %6 = insertelement <4 x float> %5, float %2, i32 3
  %7 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %6, i32 48)
  ret <4 x float> %7
}

define <4 x float> @insertps_from_broadcast_loadv4f32(<4 x float> %a, <4 x float>* nocapture readonly %b) {
; CHECK-LABEL: insertps_from_broadcast_loadv4f32:
; On X32, account for the arguments' move to registers
; X32: movl    4(%esp), %{{...}}
; CHECK-NOT: mov
; CHECK: insertps    $48
; CHECK-NEXT: ret
  %1 = load <4 x float>, <4 x float>* %b, align 4
  %2 = extractelement <4 x float> %1, i32 0
  %3 = insertelement <4 x float> undef, float %2, i32 0
  %4 = insertelement <4 x float> %3, float %2, i32 1
  %5 = insertelement <4 x float> %4, float %2, i32 2
  %6 = insertelement <4 x float> %5, float %2, i32 3
  %7 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %6, i32 48)
  ret <4 x float> %7
}

;; FIXME: We're emitting an extraneous pshufd/vbroadcast.
define <4 x float> @insertps_from_broadcast_multiple_use(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d, float* nocapture readonly %fb, i64 %index) {
; CHECK-LABEL: insertps_from_broadcast_multiple_use:
; On X32, account for the arguments' move to registers
; X32: movl    8(%esp), %eax
; X32: movl    4(%esp), %ecx
; CHECK: vbroadcastss
; CHECK-NOT: mov
; CHECK: insertps    $48
; CHECK: insertps    $48
; CHECK: vaddps
; CHECK: insertps    $48
; CHECK: insertps    $48
; CHECK: vaddps
; CHECK: vaddps
; CHECK-NEXT: ret
  %1 = getelementptr inbounds float, float* %fb, i64 %index
  %2 = load float, float* %1, align 4
  %3 = insertelement <4 x float> undef, float %2, i32 0
  %4 = insertelement <4 x float> %3, float %2, i32 1
  %5 = insertelement <4 x float> %4, float %2, i32 2
  %6 = insertelement <4 x float> %5, float %2, i32 3
  %7 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %a, <4 x float> %6, i32 48)
  %8 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %b, <4 x float> %6, i32 48)
  %9 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %c, <4 x float> %6, i32 48)
  %10 = tail call <4 x float> @llvm.x86.sse41.insertps(<4 x float> %d, <4 x float> %6, i32 48)
  %11 = fadd <4 x float> %7, %8
  %12 = fadd <4 x float> %9, %10
  %13 = fadd <4 x float> %11, %12
  ret <4 x float> %13
}
