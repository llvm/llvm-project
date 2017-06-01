; RUN: llc < %s -mtriple=aarch64-linux-eabi -o - | FileCheck %s

%struct.a= type { i64, i64, i64, i64 }

; DAG combine will try to perform a transformation that  creates a vcvtfp2fxs
; with a v4f64 input. Since v4i64 is not legal we should bail out. We can
; pottentially still create the vcvtfp2fxs node after legalization (but on a
; v2f64).

; CHECK-LABEL: fun1
define void @fun1() local_unnamed_addr {
entry:
  %mul = fmul <4 x double> zeroinitializer, <double 6.553600e+04, double 6.553600e+04, double 6.553600e+04, double 6.553600e+04>
  %toi = fptosi <4 x double> %mul to <4 x i64>
  %ptr = getelementptr inbounds %struct.a, %struct.a* undef, i64 0, i32 2
  %elem = extractelement <4 x i64> %toi, i32 1
  store i64 %elem, i64* %ptr, align 8
  call void @llvm.trap()
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @llvm.trap()

