; RUN: opt -S < %s | FileCheck %s
; RUN: llvm-dis <  %s.bc | FileCheck %s

target triple = "aarch64-linux"

define void @ftmopa_za16_nxv16i8(<vscale x 16 x i8> %zn1, <vscale x 16 x i8> %zn2, <vscale x 16 x i8> %zm, <vscale x 16 x i8> %zk) #0 {
; CHECK-LABEL: @ftmopa_za16_nxv16i8
; CHECK: call void @llvm.aarch64.sme.fp8.ftmopa.za16(i32 0, <vscale x 16 x i8> %zn1, <vscale x 16 x i8> %zn2, <vscale x 16 x i8> %zm, <vscale x 16 x i8> %zk, i32 0)
  call void @llvm.aarch64.sme.ftmopa.za16.nxv16i8(i32 0, <vscale x 16 x i8> %zn1, <vscale x 16 x i8> %zn2, <vscale x 16 x i8> %zm, <vscale x 16 x i8> %zk, i32 0)
  ret void
}

define void @ftmopa_za32_nxv16i8(<vscale x 16 x i8> %zn1, <vscale x 16 x i8> %zn2, <vscale x 16 x i8> %zm, <vscale x 16 x i8> %zk) #0 {
; CHECK-LABEL: @ftmopa_za32_nxv16i
; CHECK: call void @llvm.aarch64.sme.fp8.ftmopa.za32(i32 0, <vscale x 16 x i8> %zn1, <vscale x 16 x i8> %zn2, <vscale x 16 x i8> %zm, <vscale x 16 x i8> %zk, i32 0)
  call void @llvm.aarch64.sme.ftmopa.za32.nxv16i8(i32 0, <vscale x 16 x i8> %zn1, <vscale x 16 x i8> %zn2, <vscale x 16 x i8> %zm, <vscale x 16 x i8> %zk, i32 0)
  ret void
}


attributes #0 = {nounwind "target-features" = "+sme2,+sme-tmop,+sme-f8f16,+sme-f8f32" }
