; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

%complex = type { { double, double } }

; Function Attrs: argmemonly nounwind readonly
declare { <vscale x 2 x double>, <vscale x 2 x double> } @llvm.aarch64.sve.ld2.sret.nxv2f64(<vscale x 2 x i1>, ptr) #3

; Function Attrs: nounwind readnone
declare double @llvm.aarch64.sve.faddv.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>) #2

define void @foo1(ptr %outval, <vscale x 2 x i1> %pred, ptr %inptr) {
; CHECK-LABEL: foo1:
; CHECK: ld2d { z0.d, z1.d }, p0/z, [x1]
; CHECK-NEXT: faddv d2, p0, z0.d
; CHECK-NEXT: faddv d0, p0, z1.d
; CHECK-NEXT: mov v2.d[1], v0.d[0]
; CHECK-NEXT: str q2, [x0]
  %imagp = getelementptr inbounds %complex, ptr %outval, i64 0, i32 0, i32 1
  %1 = call { <vscale x 2 x double>, <vscale x 2 x double> } @llvm.aarch64.sve.ld2.sret.nxv2f64(<vscale x 2 x i1> %pred, ptr nonnull %inptr)
  %2 = extractvalue { <vscale x 2 x double>, <vscale x 2 x double> } %1, 0
  %3 = call double @llvm.aarch64.sve.faddv.nxv2f64(<vscale x 2 x i1> %pred, <vscale x 2 x double> %2)
  %4 = extractvalue { <vscale x 2 x double>, <vscale x 2 x double> } %1, 1
  %5 = call double @llvm.aarch64.sve.faddv.nxv2f64(<vscale x 2 x i1> %pred, <vscale x 2 x double> %4)
  store double %3, ptr %outval, align 8
  store double %5, ptr %imagp, align 8
  ret void
}

