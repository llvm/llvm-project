; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>*, i32, <2 x i1>, <2 x double>)

define <2 x double> @masked_load(<2 x i1> %mask, <2 x double>* %addr, <2 x double> %dst) {
  ; CHECK: masked_load: alignment must be a power of 2
  ; CHECK-NEXT: %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %addr, i32 3, <2 x i1> %mask, <2 x double> %dst)
  %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %addr, i32 3, <2 x i1>%mask, <2 x double> %dst)
  ret <2 x double> %res
}
