; RUN: opt < %s -msan -msan-check-access-address=0 -S | FileCheck %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=1 -S | FileCheck %s --check-prefixes=CHECK,CHECK-ORIGIN
; RUN: opt < %s -msan -msan-check-access-address=1 -S | FileCheck %s --check-prefix=ADDR

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.masked.store.v4i64.p0v4i64(<4 x i64>, <4 x i64>*, i32, <4 x i1>)
declare <4 x double> @llvm.masked.load.v4f64.p0v4f64(<4 x double>*, i32, <4 x i1>, <4 x double>)

define void @Store(<4 x i64>* %p, <4 x i64> %v, <4 x i1> %mask) sanitize_memory {
entry:
  tail call void @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %v, <4 x i64>* %p, i32 1, <4 x i1> %mask)
  ret void
}

; CHECK-LABEL: @Store(
; CHECK: %[[A:.*]] = load <4 x i64>, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-ORIGIN: %[[O:.*]] = load i32, {{.*}}@__msan_param_origin_tls to i64), i64 8)
; CHECK: %[[B:.*]] = ptrtoint <4 x i64>* %p to i64
; CHECK: %[[C:.*]] = xor i64 %[[B]], 87960930222080
; CHECK: %[[D:.*]] = inttoptr i64 %[[C]] to <4 x i64>*
; CHECK-ORIGIN: %[[E:.*]] = add i64 %[[C]], 17592186044416
; CHECK-ORIGIN: %[[F:.*]] = and i64 %[[E]], -4
; CHECK-ORIGIN: %[[G:.*]] = inttoptr i64 %[[F]] to i32*
; CHECK: call void @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %[[A]], <4 x i64>* %[[D]], i32 1, <4 x i1> %mask)
; CHECK-ORIGIN: store i32 %[[O]], i32* %[[G]], align 4
; CHECK-ORIGIN: getelementptr i32, i32* %[[G]], i32 1
; CHECK-ORIGIN: store i32 %[[O]], i32* {{.*}}, align 4
; CHECK-ORIGIN: getelementptr i32, i32* %[[G]], i32 2
; CHECK-ORIGIN: store i32 %[[O]], i32* {{.*}}, align 4
; CHECK-ORIGIN: getelementptr i32, i32* %[[G]], i32 3
; CHECK-ORIGIN: store i32 %[[O]], i32* {{.*}}, align 4
; CHECK-ORIGIN: getelementptr i32, i32* %[[G]], i32 4
; CHECK-ORIGIN: store i32 %[[O]], i32* {{.*}}, align 4
; CHECK-ORIGIN: getelementptr i32, i32* %[[G]], i32 5
; CHECK-ORIGIN: store i32 %[[O]], i32* {{.*}}, align 4
; CHECK-ORIGIN: getelementptr i32, i32* %[[G]], i32 6
; CHECK-ORIGIN: store i32 %[[O]], i32* {{.*}}, align 4
; CHECK-ORIGIN: getelementptr i32, i32* %[[G]], i32 7
; CHECK-ORIGIN: store i32 %[[O]], i32* {{.*}}, align 4
; CHECK: tail call void @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %v, <4 x i64>* %p, i32 1, <4 x i1> %mask)
; CHECK: ret void

; ADDR-LABEL: @Store(
; ADDR: %[[MASKSHADOW:.*]] = load <4 x i1>, {{.*}}@__msan_param_tls to i64), i64 40)
; ADDR: %[[ADDRSHADOW:.*]] = load i64, {{.*}}[100 x i64]* @__msan_param_tls, i32 0, i32 0)

; ADDR: %[[ADDRBAD:.*]] = icmp ne i64 %[[ADDRSHADOW]], 0
; ADDR: br i1 %[[ADDRBAD]], label {{.*}}, label {{.*}}
; ADDR: call void @__msan_warning_noreturn()

; ADDR: %[[MASKSHADOWFLAT:.*]] = bitcast <4 x i1> %[[MASKSHADOW]] to i4
; ADDR: %[[MASKBAD:.*]] = icmp ne i4 %[[MASKSHADOWFLAT]], 0
; ADDR: br i1 %[[MASKBAD]], label {{.*}}, label {{.*}}
; ADDR: call void @__msan_warning_noreturn()

; ADDR: tail call void @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %v, <4 x i64>* %p, i32 1, <4 x i1> %mask)
; ADDR: ret void


define <4 x double> @Load(<4 x double>* %p, <4 x double> %v, <4 x i1> %mask) sanitize_memory {
entry:
  %x = call <4 x double> @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %p, i32 1, <4 x i1> %mask, <4 x double> %v)
  ret <4 x double> %x
}

; CHECK-LABEL: @Load(
; CHECK: %[[A:.*]] = load <4 x i64>, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-ORIGIN: %[[O:.*]] = load i32, {{.*}}@__msan_param_origin_tls to i64), i64 8)
; CHECK: %[[B:.*]] = ptrtoint <4 x double>* %p to i64
; CHECK: %[[C:.*]] = xor i64 %[[B]], 87960930222080
; CHECK: %[[D:.*]] = inttoptr i64 %[[C]] to <4 x i64>*
; CHECK-ORIGIN: %[[E:.*]] = add i64 %[[C]], 17592186044416
; CHECK-ORIGIN: %[[F:.*]] = and i64 %[[E]], -4
; CHECK-ORIGIN: %[[G:.*]] = inttoptr i64 %[[F]] to i32*
; CHECK: %[[E:.*]] = call <4 x i64> @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %[[D]], i32 1, <4 x i1> %mask, <4 x i64> %[[A]])
; CHECK-ORIGIN: %[[H:.*]] = load i32, i32* %[[G]]
; CHECK-ORIGIN: %[[O2:.*]] = select i1 %{{.*}}, i32 %[[O]], i32 %[[H]]
; CHECK: %[[X:.*]] = call <4 x double> @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %p, i32 1, <4 x i1> %mask, <4 x double> %v)
; CHECK: store <4 x i64> %[[E]], {{.*}}@__msan_retval_tls
; CHECK-ORIGIN: store i32 %[[O2]], i32* @__msan_retval_origin_tls
; CHECK: ret <4 x double> %[[X]]

; ADDR-LABEL: @Load(
; ADDR: %[[MASKSHADOW:.*]] = load <4 x i1>, {{.*}}@__msan_param_tls to i64), i64 40)
; ADDR: %[[ADDRSHADOW:.*]] = load i64, {{.*}}[100 x i64]* @__msan_param_tls, i32 0, i32 0)

; ADDR: %[[ADDRBAD:.*]] = icmp ne i64 %[[ADDRSHADOW]], 0
; ADDR: br i1 %[[ADDRBAD]], label {{.*}}, label {{.*}}
; ADDR: call void @__msan_warning_noreturn()

; ADDR: %[[MASKSHADOWFLAT:.*]] = bitcast <4 x i1> %[[MASKSHADOW]] to i4
; ADDR: %[[MASKBAD:.*]] = icmp ne i4 %[[MASKSHADOWFLAT]], 0
; ADDR: br i1 %[[MASKBAD]], label {{.*}}, label {{.*}}
; ADDR: call void @__msan_warning_noreturn()

; ADDR: = call <4 x double> @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %p, i32 1, <4 x i1> %mask, <4 x double> %v)
; ADDR: ret <4 x double>

define void @StoreNoSanitize(<4 x i64>* %p, <4 x i64> %v, <4 x i1> %mask) {
entry:
  tail call void @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %v, <4 x i64>* %p, i32 1, <4 x i1> %mask)
  ret void
}

; CHECK-LABEL: @StoreNoSanitize(
; CHECK: %[[B:.*]] = ptrtoint <4 x i64>* %p to i64
; CHECK: %[[C:.*]] = xor i64 %[[B]], 87960930222080
; CHECK: %[[D:.*]] = inttoptr i64 %[[C]] to <4 x i64>*
; CHECK: call void @llvm.masked.store.v4i64.p0v4i64(<4 x i64> zeroinitializer, <4 x i64>* %[[D]], i32 1, <4 x i1> %mask)
; CHECK: tail call void @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %v, <4 x i64>* %p, i32 1, <4 x i1> %mask)
; CHECK: ret void

define <4 x double> @LoadNoSanitize(<4 x double>* %p, <4 x double> %v, <4 x i1> %mask) {
entry:
  %x = call <4 x double> @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %p, i32 1, <4 x i1> %mask, <4 x double> %v)
  ret <4 x double> %x
}

; CHECK-LABEL: @LoadNoSanitize(
; CHECK: %[[X:.*]] = call <4 x double> @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %p, i32 1, <4 x i1> %mask, <4 x double> %v)
; CHECK: store <4 x i64> zeroinitializer, {{.*}}@__msan_retval_tls to <4 x i64>*)
; CHECK: ret <4 x double> %[[X]]
