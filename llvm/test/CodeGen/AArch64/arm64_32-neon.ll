; RUN: llc -mtriple=arm64_32-apple-ios7.0 -mcpu=cyclone %s -o - | FileCheck %s

define <2 x double> @test_insert_elt(<2 x double> %vec, double %val) {
; CHECK-LABEL: test_insert_elt:
; CHECK: mov.d v0[0], v1[0]
  %res = insertelement <2 x double> %vec, double %val, i32 0
  ret <2 x double> %res
}

define void @test_split_16B(<4 x float> %val, ptr %addr) {
; CHECK-LABEL: test_split_16B:
; CHECK: str q0, [x0]
  store <4 x float> %val, ptr %addr, align 8
  ret void
}

define void @test_split_16B_splat(<4 x i32>, ptr %addr) {
; CHECK-LABEL: test_split_16B_splat:
; CHECK: str {{q[0-9]+}}

  %vec.tmp0 = insertelement <4 x i32> undef, i32 42, i32 0
  %vec.tmp1 = insertelement <4 x i32> %vec.tmp0, i32 42, i32 1
  %vec.tmp2 = insertelement <4 x i32> %vec.tmp1, i32 42, i32 2
  %vec = insertelement <4 x i32> %vec.tmp2, i32 42, i32 3

  store <4 x i32> %vec, ptr %addr, align 8
  ret void
}


%vec = type <2 x double>

declare {%vec, %vec} @llvm.aarch64.neon.ld2r.v2f64.p0(ptr)
define {%vec, %vec} @test_neon_load(ptr %addr) {
; CHECK-LABEL: test_neon_load:
; CHECK: ld2r.2d { v0, v1 }, [x0]
  %res = call {%vec, %vec} @llvm.aarch64.neon.ld2r.v2f64.p0(ptr %addr)
  ret {%vec, %vec} %res
}

declare {%vec, %vec} @llvm.aarch64.neon.ld2lane.v2f64.p0(%vec, %vec, i64, ptr)
define {%vec, %vec} @test_neon_load_lane(ptr %addr, %vec %in1, %vec %in2) {
; CHECK-LABEL: test_neon_load_lane:
; CHECK: ld2.d { v0, v1 }[0], [x0]
  %res = call {%vec, %vec} @llvm.aarch64.neon.ld2lane.v2f64.p0(%vec %in1, %vec %in2, i64 0, ptr %addr)
  ret {%vec, %vec} %res
}

declare void @llvm.aarch64.neon.st2.v2f64.p0(%vec, %vec, ptr)
define void @test_neon_store(ptr %addr, %vec %in1, %vec %in2) {
; CHECK-LABEL: test_neon_store:
; CHECK: st2.2d { v0, v1 }, [x0]
  call void @llvm.aarch64.neon.st2.v2f64.p0(%vec %in1, %vec %in2, ptr %addr)
  ret void
}

declare void @llvm.aarch64.neon.st2lane.v2f64.p0(%vec, %vec, i64, ptr)
define void @test_neon_store_lane(ptr %addr, %vec %in1, %vec %in2) {
; CHECK-LABEL: test_neon_store_lane:
; CHECK: st2.d { v0, v1 }[1], [x0]
  call void @llvm.aarch64.neon.st2lane.v2f64.p0(%vec %in1, %vec %in2, i64 1, ptr %addr)
  ret void
}

declare {%vec, %vec} @llvm.aarch64.neon.ld2.v2f64.p0(ptr)
define {{%vec, %vec}, ptr} @test_neon_load_post(ptr %addr, i32 %offset) {
; CHECK-LABEL: test_neon_load_post:
; CHECK-DAG: sxtw [[OFFSET:x[0-9]+]], w1
; CHECK: ld2.2d { v0, v1 }, [x0], [[OFFSET]]

  %vecs = call {%vec, %vec} @llvm.aarch64.neon.ld2.v2f64.p0(ptr %addr)

  %addr.new = getelementptr inbounds i8, ptr %addr, i32 %offset

  %res.tmp = insertvalue {{%vec, %vec}, ptr} undef, {%vec, %vec} %vecs, 0
  %res = insertvalue {{%vec, %vec}, ptr} %res.tmp, ptr %addr.new, 1
  ret {{%vec, %vec}, ptr} %res
}

define {{%vec, %vec}, ptr} @test_neon_load_post_lane(ptr %addr, i32 %offset, %vec %in1, %vec %in2) {
; CHECK-LABEL: test_neon_load_post_lane:
; CHECK-DAG: sxtw [[OFFSET:x[0-9]+]], w1
; CHECK: ld2.d { v0, v1 }[1], [x0], [[OFFSET]]

  %vecs = call {%vec, %vec} @llvm.aarch64.neon.ld2lane.v2f64.p0(%vec %in1, %vec %in2, i64 1, ptr %addr)

  %addr.new = getelementptr inbounds i8, ptr %addr, i32 %offset

  %res.tmp = insertvalue {{%vec, %vec}, ptr} undef, {%vec, %vec} %vecs, 0
  %res = insertvalue {{%vec, %vec}, ptr} %res.tmp, ptr %addr.new, 1
  ret {{%vec, %vec}, ptr} %res
}

define ptr @test_neon_store_post(ptr %addr, i32 %offset, %vec %in1, %vec %in2) {
; CHECK-LABEL: test_neon_store_post:
; CHECK-DAG: sxtw [[OFFSET:x[0-9]+]], w1
; CHECK: st2.2d { v0, v1 }, [x0], [[OFFSET]]

  call void @llvm.aarch64.neon.st2.v2f64.p0(%vec %in1, %vec %in2, ptr %addr)

  %addr.new = getelementptr inbounds i8, ptr %addr, i32 %offset

  ret ptr %addr.new
}

define ptr @test_neon_store_post_lane(ptr %addr, i32 %offset, %vec %in1, %vec %in2) {
; CHECK-LABEL: test_neon_store_post_lane:
; CHECK: sxtw [[OFFSET:x[0-9]+]], w1
; CHECK: st2.d { v0, v1 }[0], [x0], [[OFFSET]]

  call void @llvm.aarch64.neon.st2lane.v2f64.p0(%vec %in1, %vec %in2, i64 0, ptr %addr)

  %addr.new = getelementptr inbounds i8, ptr %addr, i32 %offset

  ret ptr %addr.new
}

; ld1 is slightly different because it goes via ISelLowering of normal IR ops
; rather than an intrinsic.
define {%vec, ptr} @test_neon_ld1_post_lane(ptr %addr, i32 %offset, %vec %in) {
; CHECK-LABEL: test_neon_ld1_post_lane:
; CHECK: sbfiz [[OFFSET:x[0-9]+]], x1, #3, #32
; CHECK: ld1.d { v0 }[0], [x0], [[OFFSET]]

  %loaded = load double, ptr %addr, align 8
  %newvec = insertelement %vec %in, double %loaded, i32 0

  %addr.new = getelementptr inbounds double, ptr %addr, i32 %offset

  %res.tmp = insertvalue {%vec, ptr} undef, %vec %newvec, 0
  %res = insertvalue {%vec, ptr} %res.tmp, ptr %addr.new, 1

  ret {%vec, ptr} %res
}

define {{%vec, %vec}, ptr} @test_neon_load_post_exact(ptr %addr) {
; CHECK-LABEL: test_neon_load_post_exact:
; CHECK: ld2.2d { v0, v1 }, [x0], #32

  %vecs = call {%vec, %vec} @llvm.aarch64.neon.ld2.v2f64.p0(ptr %addr)

  %addr.new = getelementptr inbounds i8, ptr %addr, i32 32

  %res.tmp = insertvalue {{%vec, %vec}, ptr} undef, {%vec, %vec} %vecs, 0
  %res = insertvalue {{%vec, %vec}, ptr} %res.tmp, ptr %addr.new, 1
  ret {{%vec, %vec}, ptr} %res
}

define {%vec, ptr} @test_neon_ld1_post_lane_exact(ptr %addr, %vec %in) {
; CHECK-LABEL: test_neon_ld1_post_lane_exact:
; CHECK: ld1.d { v0 }[0], [x0], #8

  %loaded = load double, ptr %addr, align 8
  %newvec = insertelement %vec %in, double %loaded, i32 0

  %addr.new = getelementptr inbounds double, ptr %addr, i32 1

  %res.tmp = insertvalue {%vec, ptr} undef, %vec %newvec, 0
  %res = insertvalue {%vec, ptr} %res.tmp, ptr %addr.new, 1

  ret {%vec, ptr} %res
}

; As in the general load/store case, this GEP has defined semantics when the
; address wraps. We cannot use post-indexed addressing.
define {%vec, ptr} @test_neon_ld1_notpost_lane_exact(ptr %addr, %vec %in) {
; CHECK-LABEL: test_neon_ld1_notpost_lane_exact:
; CHECK-NOT: ld1.d { {{v[0-9]+}} }[0], [{{x[0-9]+|sp}}], #8
; CHECK: add w0, w0, #8
; CHECK: ret

  %loaded = load double, ptr %addr, align 8
  %newvec = insertelement %vec %in, double %loaded, i32 0

  %addr.new = getelementptr double, ptr %addr, i32 1

  %res.tmp = insertvalue {%vec, ptr} undef, %vec %newvec, 0
  %res = insertvalue {%vec, ptr} %res.tmp, ptr %addr.new, 1

  ret {%vec, ptr} %res
}

define {%vec, ptr} @test_neon_ld1_notpost_lane(ptr %addr, i32 %offset, %vec %in) {
; CHECK-LABEL: test_neon_ld1_notpost_lane:
; CHECK-NOT: ld1.d { {{v[0-9]+}} }[0], [{{x[0-9]+|sp}}], {{x[0-9]+|sp}}
; CHECK: add w0, w0, w1, lsl #3
; CHECK: ret

  %loaded = load double, ptr %addr, align 8
  %newvec = insertelement %vec %in, double %loaded, i32 0

  %addr.new = getelementptr double, ptr %addr, i32 %offset

  %res.tmp = insertvalue {%vec, ptr} undef, %vec %newvec, 0
  %res = insertvalue {%vec, ptr} %res.tmp, ptr %addr.new, 1

  ret {%vec, ptr} %res
}
