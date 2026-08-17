; RUN: not opt -passes=verify --disable-output %s 2>&1 | FileCheck %s


; Casts
; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <8 x float>
; CHECK-NEXT: declare <8 x float> @llvm.vp.fptoui.v8f32.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fptoui.v8f32.v8f32(<8 x float>, <8 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any fp vector, but got <8 x i32>
; CHECK-NEXT: declare <8 x i32> @llvm.vp.fptoui.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.fptoui.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <8 x float>
; CHECK-NEXT: declare <8 x float> @llvm.vp.fptosi.v8f32.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.fptosi.v8f32.v8f32(<8 x float>, <8 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any fp vector, but got <8 x i32>
; CHECK-NEXT: declare <8 x i32> @llvm.vp.fptosi.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.fptosi.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any fp vector, but got <8 x i32>
; CHECK-NEXT: declare <8 x i32> @llvm.vp.uitofp.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.uitofp.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got <8 x float>
; CHECK-NEXT: declare <8 x float> @llvm.vp.uitofp.v8f32.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.uitofp.v8f32.v8f32(<8 x float>, <8 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any fp vector, but got <8 x i32>
; CHECK-NEXT: declare <8 x i32> @llvm.vp.sitofp.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.sitofp.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got <8 x float>
; CHECK-NEXT: declare <8 x float> @llvm.vp.sitofp.v8f32.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.sitofp.v8f32.v8f32(<8 x float>, <8 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any fp vector, but got <8 x i32>
; CHECK-NEXT: declare <8 x i32> @llvm.vp.fptrunc.v8i32.v8f64(<8 x double>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.fptrunc.v8i32.v8f64(<8 x double>, <8 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any fp vector, but got <8 x i32>
; CHECK-NEXT: declare <8 x double> @llvm.vp.fpext.v8f64.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x double> @llvm.vp.fpext.v8f64.v8i32(<8 x i32>, <8 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <8 x float>
; CHECK-NEXT: declare <8 x float> @llvm.vp.trunc.v8i32.v8i64(<8 x i64>, <8 x i1>, i32)
declare <8 x float> @llvm.vp.trunc.v8i32.v8i64(<8 x i64>, <8 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got <8 x float>
; CHECK-NEXT: declare <8 x i64> @llvm.vp.zext.v8i64.v8f32(<8 x float>, <8 x i1>, i32)
declare <8 x i64> @llvm.vp.zext.v8i64.v8f32(<8 x float>, <8 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <8 x double>
; CHECK-NEXT: declare <8 x double> @llvm.vp.sext.v8f64.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x double> @llvm.vp.sext.v8f64.v8i32(<8 x i32>, <8 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any pointer vector, but got <8 x i32>
; CHECK-NEXT: declare <8 x i32> @llvm.vp.ptrtoint.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.ptrtoint.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any pointer vector, but got ptr
; CHECK-NEXT: declare <8 x i32> @llvm.vp.ptrtoint.v8i32.p0(ptr, <8 x i1>, i32)
declare <8 x i32> @llvm.vp.ptrtoint.v8i32.p0(ptr, <8 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any pointer vector, but got ptr
; CHECK-NEXT: declare ptr @llvm.vp.inttoptr.p0.v8i32(<8 x i32>, <8 x i1>, i32)
declare ptr @llvm.vp.inttoptr.p0.v8i32(<8 x i32>, <8 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any pointer vector, but got <8 x i32>
; CHECK-NEXT: declare <8 x i32> @llvm.vp.inttoptr.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)
declare <8 x i32>  @llvm.vp.inttoptr.v8i32.v8i32(<8 x i32>, <8 x i1>, i32)

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with 7 elements (overload type 0 is <7 x ptr>), but got <8 x i1>
; CHECK-NEXT: declare <7 x ptr> @llvm.vp.inttoptr.v7p0.v8i32(<8 x i32>, <8 x i1>, i32)
declare <7 x ptr>  @llvm.vp.inttoptr.v7p0.v8i32(<8 x i32>, <8 x i1>, i32)
