; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

;; Scalar Integer Compare

define i64 @test_vceqd(i64 %a, i64 %b) {
; CHECK: test_vceqd
; CHECK: cmeq {{d[0-9]+}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vceq.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vceq1.i = insertelement <1 x i64> undef, i64 %b, i32 0
  %vceq2.i = call <1 x i64> @llvm.aarch64.neon.vceq.v1i64.v1i64.v1i64(<1 x i64> %vceq.i, <1 x i64> %vceq1.i)
  %0 = extractelement <1 x i64> %vceq2.i, i32 0
  ret i64 %0
}

define i64 @test_vceqzd(i64 %a) {
; CHECK: test_vceqzd
; CHECK: cmeq {{d[0-9]}}, {{d[0-9]}}, #0x0
entry:
  %vceqz.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vceqz1.i = call <1 x i64> @llvm.aarch64.neon.vceq.v1i64.v1i64.v1i64(<1 x i64> %vceqz.i, <1 x i64> zeroinitializer)
  %0 = extractelement <1 x i64> %vceqz1.i, i32 0
  ret i64 %0
}

define i64 @test_vcged(i64 %a, i64 %b) {
; CHECK: test_vcged
; CHECK: cmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcge.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vcge1.i = insertelement <1 x i64> undef, i64 %b, i32 0
  %vcge2.i = call <1 x i64> @llvm.aarch64.neon.vcge.v1i64.v1i64.v1i64(<1 x i64> %vcge.i, <1 x i64> %vcge1.i)
  %0 = extractelement <1 x i64> %vcge2.i, i32 0
  ret i64 %0
}

define i64 @test_vcgezd(i64 %a) {
; CHECK: test_vcgezd
; CHECK: cmge {{d[0-9]}}, {{d[0-9]}}, #0x0
entry:
  %vcgez.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vcgez1.i = call <1 x i64> @llvm.aarch64.neon.vcge.v1i64.v1i64.v1i64(<1 x i64> %vcgez.i, <1 x i64> zeroinitializer)
  %0 = extractelement <1 x i64> %vcgez1.i, i32 0
  ret i64 %0
}

define i64 @test_vcgtd(i64 %a, i64 %b) {
; CHECK: test_vcgtd
; CHECK: cmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcgt.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vcgt1.i = insertelement <1 x i64> undef, i64 %b, i32 0
  %vcgt2.i = call <1 x i64> @llvm.aarch64.neon.vcgt.v1i64.v1i64.v1i64(<1 x i64> %vcgt.i, <1 x i64> %vcgt1.i)
  %0 = extractelement <1 x i64> %vcgt2.i, i32 0
  ret i64 %0
}

define i64 @test_vcgtzd(i64 %a) {
; CHECK: test_vcgtzd
; CHECK: cmgt {{d[0-9]}}, {{d[0-9]}}, #0x0
entry:
  %vcgtz.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vcgtz1.i = call <1 x i64> @llvm.aarch64.neon.vcgt.v1i64.v1i64.v1i64(<1 x i64> %vcgtz.i, <1 x i64> zeroinitializer)
  %0 = extractelement <1 x i64> %vcgtz1.i, i32 0
  ret i64 %0
}

define i64 @test_vcled(i64 %a, i64 %b) {
; CHECK: test_vcled
; CHECK: cmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcgt.i = insertelement <1 x i64> undef, i64 %b, i32 0
  %vcgt1.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vcgt2.i = call <1 x i64> @llvm.aarch64.neon.vcgt.v1i64.v1i64.v1i64(<1 x i64> %vcgt.i, <1 x i64> %vcgt1.i)
  %0 = extractelement <1 x i64> %vcgt2.i, i32 0
  ret i64 %0
}

define i64 @test_vclezd(i64 %a) {
; CHECK: test_vclezd
; CHECK: cmle {{d[0-9]}}, {{d[0-9]}}, #0x0
entry:
  %vclez.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vclez1.i = call <1 x i64> @llvm.aarch64.neon.vclez.v1i64.v1i64.v1i64(<1 x i64> %vclez.i, <1 x i64> zeroinitializer)
  %0 = extractelement <1 x i64> %vclez1.i, i32 0
  ret i64 %0
}

define i64 @test_vcltd(i64 %a, i64 %b) {
; CHECK: test_vcltd
; CHECK: cmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcge.i = insertelement <1 x i64> undef, i64 %b, i32 0
  %vcge1.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vcge2.i = call <1 x i64> @llvm.aarch64.neon.vcge.v1i64.v1i64.v1i64(<1 x i64> %vcge.i, <1 x i64> %vcge1.i)
  %0 = extractelement <1 x i64> %vcge2.i, i32 0
  ret i64 %0
}

define i64 @test_vcltzd(i64 %a) {
; CHECK: test_vcltzd
; CHECK: cmlt {{d[0-9]}}, {{d[0-9]}}, #0x0
entry:
  %vcltz.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vcltz1.i = call <1 x i64> @llvm.aarch64.neon.vcltz.v1i64.v1i64.v1i64(<1 x i64> %vcltz.i, <1 x i64> zeroinitializer)
  %0 = extractelement <1 x i64> %vcltz1.i, i32 0
  ret i64 %0
}

define i64 @test_vtstd(i64 %a, i64 %b) {
; CHECK: test_vtstd
; CHECK: cmtst {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vtst.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vtst1.i = insertelement <1 x i64> undef, i64 %b, i32 0
  %vtst2.i = call <1 x i64> @llvm.aarch64.neon.vtstd.v1i64.v1i64.v1i64(<1 x i64> %vtst.i, <1 x i64> %vtst1.i)
  %0 = extractelement <1 x i64> %vtst2.i, i32 0
  ret i64 %0
}


define <1 x i64> @test_vcage_f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK: test_vcage_f64
; CHECK: facge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %vcage2.i = tail call <1 x i64> @llvm.aarch64.neon.vcage.v1i64.v1f64.v1f64(<1 x double> %a, <1 x double> %b) #2
  ret <1 x i64> %vcage2.i
}

define <1 x i64> @test_vcagt_f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK: test_vcagt_f64
; CHECK: facgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %vcagt2.i = tail call <1 x i64> @llvm.aarch64.neon.vcagt.v1i64.v1f64.v1f64(<1 x double> %a, <1 x double> %b) #2
  ret <1 x i64> %vcagt2.i
}

define <1 x i64> @test_vcale_f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK: test_vcale_f64
; CHECK: facge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %vcage2.i = tail call <1 x i64> @llvm.aarch64.neon.vcage.v1i64.v1f64.v1f64(<1 x double> %b, <1 x double> %a) #2
  ret <1 x i64> %vcage2.i
}

define <1 x i64> @test_vcalt_f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK: test_vcalt_f64
; CHECK: facgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %vcagt2.i = tail call <1 x i64> @llvm.aarch64.neon.vcagt.v1i64.v1f64.v1f64(<1 x double> %b, <1 x double> %a) #2
  ret <1 x i64> %vcagt2.i
}

define <1 x i64> @test_vceq_s64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vceq_s64
; CHECK: cmeq {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp eq <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vceq_u64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vceq_u64
; CHECK: cmeq {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp eq <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vceq_f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK: test_vceq_f64
; CHECK: fcmeq {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = fcmp oeq <1 x double> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vcge_s64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vcge_s64
; CHECK: cmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp sge <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vcge_u64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vcge_u64
; CHECK: cmhs {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp uge <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vcge_f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK: test_vcge_f64
; CHECK: fcmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = fcmp oge <1 x double> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vcle_s64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vcle_s64
; CHECK: cmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp sle <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vcle_u64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vcle_u64
; CHECK: cmhs {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp ule <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vcle_f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK: test_vcle_f64
; CHECK: fcmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = fcmp ole <1 x double> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vcgt_s64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vcgt_s64
; CHECK: cmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp sgt <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vcgt_u64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vcgt_u64
; CHECK: cmhi {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp ugt <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vcgt_f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK: test_vcgt_f64
; CHECK: fcmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = fcmp ogt <1 x double> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vclt_s64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vclt_s64
; CHECK: cmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp slt <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vclt_u64(<1 x i64> %a, <1 x i64> %b) #0 {
; CHECK: test_vclt_u64
; CHECK: cmhi {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = icmp ult <1 x i64> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vclt_f64(<1 x double> %a, <1 x double> %b) #0 {
; CHECK: test_vclt_f64
; CHECK: fcmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
  %cmp.i = fcmp olt <1 x double> %a, %b
  %sext.i = sext <1 x i1> %cmp.i to <1 x i64>
  ret <1 x i64> %sext.i
}

define <1 x i64> @test_vceqz_s64(<1 x i64> %a) #0 {
; CHECK: test_vceqz_s64
; CHECK: cmeq {{d[0-9]}}, {{d[0-9]}}, #0x0
  %1 = icmp eq <1 x i64> %a, zeroinitializer
  %vceqz.i = zext <1 x i1> %1 to <1 x i64>
  ret <1 x i64> %vceqz.i
}

define <1 x i64> @test_vceqz_u64(<1 x i64> %a) #0 {
; CHECK: test_vceqz_u64
; CHECK: cmeq {{d[0-9]}}, {{d[0-9]}}, #0x0
  %1 = icmp eq <1 x i64> %a, zeroinitializer
  %vceqz.i = zext <1 x i1> %1 to <1 x i64>
  ret <1 x i64> %vceqz.i
}

define <1 x i64> @test_vceqz_p64(<1 x i64> %a) #0 {
; CHECK: test_vceqz_p64
; CHECK: cmeq {{d[0-9]}}, {{d[0-9]}}, #0x0
  %1 = icmp eq <1 x i64> %a, zeroinitializer
  %vceqz.i = zext <1 x i1> %1 to <1 x i64>
  ret <1 x i64> %vceqz.i
}

define <2 x i64> @test_vceqzq_p64(<2 x i64> %a) #0 {
; CHECK: test_vceqzq_p64
; CHECK: cmeq  {{v[0-9]}}.2d, {{v[0-9]}}.2d, #0
  %1 = icmp eq <2 x i64> %a, zeroinitializer
  %vceqz.i = zext <2 x i1> %1 to <2 x i64>
  ret <2 x i64> %vceqz.i
}

define <1 x i64> @test_vcgez_s64(<1 x i64> %a) #0 {
; CHECK: test_vcgez_s64
; CHECK: cmge {{d[0-9]}}, {{d[0-9]}}, #0x0
  %1 = icmp sge <1 x i64> %a, zeroinitializer
  %vcgez.i = zext <1 x i1> %1 to <1 x i64>
  ret <1 x i64> %vcgez.i
}

define <1 x i64> @test_vclez_s64(<1 x i64> %a) #0 {
; CHECK: test_vclez_s64
; CHECK: cmle {{d[0-9]}}, {{d[0-9]}}, #0x0
  %1 = icmp sle <1 x i64> %a, zeroinitializer
  %vclez.i = zext <1 x i1> %1 to <1 x i64>
  ret <1 x i64> %vclez.i
}

define <1 x i64> @test_vcgtz_s64(<1 x i64> %a) #0 {
; CHECK: test_vcgtz_s64
; CHECK: cmgt {{d[0-9]}}, {{d[0-9]}}, #0x0
  %1 = icmp sgt <1 x i64> %a, zeroinitializer
  %vcgtz.i = zext <1 x i1> %1 to <1 x i64>
  ret <1 x i64> %vcgtz.i
}

define <1 x i64> @test_vcltz_s64(<1 x i64> %a) #0 {
; CHECK: test_vcltz_s64
; CHECK: cmlt {{d[0-9]}}, {{d[0-9]}}, #0
  %1 = icmp slt <1 x i64> %a, zeroinitializer
  %vcltz.i = zext <1 x i1> %1 to <1 x i64>
  ret <1 x i64> %vcltz.i
}

declare <1 x i64> @llvm.aarch64.neon.vcagt.v1i64.v1f64.v1f64(<1 x double>, <1 x double>)
declare <1 x i64> @llvm.aarch64.neon.vcage.v1i64.v1f64.v1f64(<1 x double>, <1 x double>)
declare <1 x i64> @llvm.aarch64.neon.vtstd.v1i64.v1i64.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vcltz.v1i64.v1i64.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vchs.v1i64.v1i64.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vcge.v1i64.v1i64.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vclez.v1i64.v1i64.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vchi.v1i64.v1i64.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vcgt.v1i64.v1i64.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vceq.v1i64.v1i64.v1i64(<1 x i64>, <1 x i64>)
