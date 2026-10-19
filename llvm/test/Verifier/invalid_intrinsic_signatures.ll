; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: intrinsic return type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.alloc.token.id.v4i32(metadata)
declare <4 x i32> @llvm.alloc.token.id.v4i32(metadata)

; CHECK: intrinsic return type (overload type 0) expected any pointer type, but got <2 x ptr>
; CHECK-NEXT: declare <2 x ptr> @llvm.returnaddress(i32)
declare <2 x ptr> @llvm.returnaddress(i32)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare token @llvm.coro.alloca.alloc.v4i32(<4 x i32>, i32)
declare token @llvm.coro.alloca.alloc.v4i32(<4 x i32>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.experimental.cttz.elts.v4i32.v4i32(<4 x i32>, i1)
declare <4 x i32> @llvm.experimental.cttz.elts.v4i32.v4i32(<4 x i32>, i1)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got i32
; CHECK-NEXT: declare i32 @llvm.experimental.cttz.elts.i32.i32(i32, i1)
declare i32 @llvm.experimental.cttz.elts.i32.i32(i32, i1)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare i32 @llvm.experimental.cttz.elts.i32.v4f32(<4 x float>, i1)
declare i32 @llvm.experimental.cttz.elts.i32.v4f32(<4 x float>, i1)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare ptr @llvm.load.relative.v4i32(ptr, <4 x i32>)
declare ptr @llvm.load.relative.v4i32(ptr, <4 x i32>)

; CHECK: intrinsic return type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.objectsize.v4i32.p0(ptr, i1, i1, i1)
declare <4 x i32> @llvm.objectsize.v4i32.p0(ptr, i1, i1, i1)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x float> @llvm.powi.v4f32.v4i32(<4 x float>, <4 x i32>)
declare <4 x float> @llvm.powi.v4f32.v4i32(<4 x float>, <4 x i32>)

; CHECK: intrinsic return type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.vscale.v4i32()
declare <4 x i32> @llvm.vscale.v4i32()

; CHECK: intrinsic argument 0 type (overload type 0) expected any pointer vector, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.add.v4i32.i32(<4 x i32>, i32, <4 x i1>)
declare void @llvm.experimental.vector.histogram.add.v4i32.i32(<4 x i32>, i32, <4 x i1>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any pointer vector, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.uadd.sat.v4i32.i32(<4 x i32>, i32, <4 x i1>)
declare void @llvm.experimental.vector.histogram.uadd.sat.v4i32.i32(<4 x i32>, i32, <4 x i1>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any pointer vector, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.umin.v4i32.i32(<4 x i32>, i32, <4 x i1>)
declare void @llvm.experimental.vector.histogram.umin.v4i32.i32(<4 x i32>, i32, <4 x i1>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any pointer vector, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.umax.v4i32.i32(<4 x i32>, i32, <4 x i1>)
declare void @llvm.experimental.vector.histogram.umax.v4i32.i32(<4 x i32>, i32, <4 x i1>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.add.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)
declare void @llvm.experimental.vector.histogram.add.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.uadd.sat.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)
declare void @llvm.experimental.vector.histogram.uadd.sat.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.umin.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)
declare void @llvm.experimental.vector.histogram.umin.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.umax.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)
declare void @llvm.experimental.vector.histogram.umax.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memcpy.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)
declare void @llvm.memcpy.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memcpy.inline.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)
declare void @llvm.memcpy.inline.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memmove.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)
declare void @llvm.memmove.p0.p0.v4i32(ptr, ptr, <4 x i32>, i1)

; CHECK: intrinsic argument 2 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memset.p0.v4i32(ptr, i8, <4 x i32>, i1)
declare void @llvm.memset.p0.v4i32(ptr, i8, <4 x i32>, i1)

; CHECK: intrinsic argument 2 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memset.inline.p0.v4i32(ptr, i8, <4 x i32>, i1)
declare void @llvm.memset.inline.p0.v4i32(ptr, i8, <4 x i32>, i1)

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.memset.pattern.p0.i32.v4i32(ptr, i32, <4 x i32>, i1)
declare void @llvm.experimental.memset.pattern.p0.i32.v4i32(ptr, i32, <4 x i32>, i1)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x i1> @llvm.experimental.vector.match.v4f32.v4f32(<4 x float>, <4 x float>, <4 x i1>)
declare <4 x i1> @llvm.experimental.vector.match.v4f32.v4f32(<4 x float>, <4 x float>, <4 x i1>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x i1> @llvm.experimental.vector.match.v4i32.v4f32(<4 x i32>, <4 x float>, <4 x i1>)
declare <4 x i1> @llvm.experimental.vector.match.v4i32.v4f32(<4 x i32>, <4 x float>, <4 x i1>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got i32
; CHECK-NEXT: declare i1 @llvm.experimental.vector.match.i32.i32(i32, i32, i1)
declare i1 @llvm.experimental.vector.match.i32.i32(i32, i32, i1)

; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector (overload type 0 is <4 x i32>), but got i1
; CHECK-NEXT: declare <4 x i1> @llvm.experimental.vector.match.v4i32.v4i32(<4 x i32>, <4 x i32>, i1)
declare <4 x i1> @llvm.experimental.vector.match.v4i32.v4i32(<4 x i32>, <4 x i32>, i1)

; CHECK: intrinsic argument 2 vector element type expected i1, but got i32
; CHECK-NEXT: declare <8 x i1> @llvm.experimental.vector.match.v8i32.v8i32(<8 x i32>, <8 x i32>, <8 x i32>)
declare <8 x i1> @llvm.experimental.vector.match.v8i32.v8i32(<8 x i32>, <8 x i32>, <8 x i32>)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.masked.udiv.v4f32(<4 x float>, <4 x float>, <4 x i1>)
declare <4 x float> @llvm.masked.udiv.v4f32(<4 x float>, <4 x float>, <4 x i1>)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.masked.sdiv.v4f32(<4 x float>, <4 x float>, <4 x i1>)
declare <4 x float> @llvm.masked.sdiv.v4f32(<4 x float>, <4 x float>, <4 x i1>)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.masked.urem.v4f32(<4 x float>, <4 x float>, <4 x i1>)
declare <4 x float> @llvm.masked.urem.v4f32(<4 x float>, <4 x float>, <4 x i1>)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.masked.srem.v4f32(<4 x float>, <4 x float>, <4 x i1>)
declare <4 x float> @llvm.masked.srem.v4f32(<4 x float>, <4 x float>, <4 x i1>)

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with vscale x 4 elements (overload type 0 is <vscale x 4 x i32>), but got <4 x i1>
; CHECK-NEXT: declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr, <4 x i1>, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr, <4 x i1>, <vscale x 4 x i32>)

; CHECK: intrinsic return type (overload type 0) expected any vector type, but got i32
; CHECK-NEXT: declare i32 @llvm.masked.load.i32.p0(ptr, i1, i32)
declare i32 @llvm.masked.load.i32.p0(ptr, i1, i32)

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector (overload type 0 is <4 x i32>), but got i1
; CHECK-NEXT: declare <4 x i32> @llvm.masked.load.v4i32.p0(ptr, i1, <4 x i32>)
declare <4 x i32> @llvm.masked.load.v4i32.p0(ptr, i1, <4 x i32>)

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with 8 elements (overload type 0 is <8 x i32>), but got <5 x i1>
; CHECK-NEXT: declare <8 x i32> @llvm.masked.load.v8i32.p0(ptr, <5 x i1>, <8 x i32>)
declare <8 x i32> @llvm.masked.load.v8i32.p0(ptr, <5 x i1>, <8 x i32>)

; CHECK: intrinsic argument 2 type (matching overload type 0) expected <7 x i32>, but got <6 x i32>
; CHECK-NEXT: declare <7 x i32> @llvm.masked.load.v7i32.p0(ptr, <7 x i1>, <6 x i32>)
declare <7 x i32> @llvm.masked.load.v7i32.p0(ptr, <7 x i1>, <6 x i32>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any vector type, but got i32
; CHECK-NEXT: declare void @llvm.masked.store.i32.p0(i32, ptr, i1)
declare void @llvm.masked.store.i32.p0(i32, ptr, i1)

; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector (overload type 0 is <4 x i32>), but got i1
; CHECK-NEXT: declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr, i1)
declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr, i1)

; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector with 5 elements (overload type 0 is <5 x i32>), but got <4 x i1>
; CHECK-NEXT: declare void @llvm.masked.store.v5i32.p0(<5 x i32>, ptr, <4 x i1>)
declare void @llvm.masked.store.v5i32.p0(<5 x i32>, ptr, <4 x i1>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any fp vector, but got float
; CHECK-NEXT: declare float @llvm.vector.reduce.fmax.f32(float)
declare float @llvm.vector.reduce.fmax.f32(float)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got i32
; CHECK-NEXT: declare i32 @llvm.vector.reduce.smax.i32(i32)
declare i32 @llvm.vector.reduce.smax.i32(i32)

; CHECK: intrinsic argument 0 type (vector element of overload type 0) expected float (overload type 0 is <4 x float>), but got double
; CHECK-NEXT: declare float @llvm.vector.reduce.fadd.v4f32(double, <4 x float>)
declare float @llvm.vector.reduce.fadd.v4f32(double, <4 x float>)

; CHECK: intrinsic return type (vector element of overload type 0) expected i32 (overload type 0 is <4 x i32>), but got i64
; CHECK-NEXT: declare i64 @llvm.vector.reduce.add.v4i32(<4 x i32>)
declare i64 @llvm.vector.reduce.add.v4i32(<4 x i32>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vector.reduce.umin.v4f32(<4 x float>)
declare float @llvm.vector.reduce.umin.v4f32(<4 x float>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got <4 x ptr>
; CHECK-NEXT: declare ptr @llvm.vector.reduce.or.v4p0(<4 x ptr>)
declare ptr @llvm.vector.reduce.or.v4p0(<4 x ptr>)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vector.reduce.fadd.v4i32(i32, <4 x i32>)
declare i32 @llvm.vector.reduce.fadd.v4i32(i32, <4 x i32>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any fp vector, but got <4 x ptr>
; CHECK-NEXT: declare ptr @llvm.vector.reduce.fmin.v4p0(<4 x ptr>)
declare ptr @llvm.vector.reduce.fmin.v4p0(<4 x ptr>)

; CHECK: intrinsic argument 0 type (matching overload type 0) expected i32, but got float
; CHECK-NEXT: declare i32 @llvm.sadd.sat.i32(float, i32)
declare i32 @llvm.sadd.sat.i32(float, i32)

; CHECK: intrinsic argument 1 type (matching overload type 0) expected i37, but got half
; CHECK-NEXT: declare i37 @llvm.uadd.sat.i37(i37, half)
declare i37 @llvm.uadd.sat.i37(i37, half)

; CHECK: intrinsic argument 0 type (matching overload type 0) expected <4 x i32>, but got <5 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.ssub.sat.v4i32(<5 x i32>, <4 x i32>)
declare <4 x i32> @llvm.ssub.sat.v4i32(<5 x i32>, <4 x i32>)

; CHECK: intrinsic argument 1 type (matching overload type 0) expected <3 x i37>, but got <3 x i32>
; CHECK-NEXT: declare <3 x i37> @llvm.usub.sat.v3i37(<3 x i37>, <3 x i32>)
declare <3 x i37> @llvm.usub.sat.v3i37(<3 x i37>, <3 x i32>)

; CHECK: intrinsic return type (overload type 0) expected any integer or integer vector, but got float
; CHECK-NEXT: declare float @llvm.ushl.sat.i32(i32, i32)
declare float @llvm.ushl.sat.i32(i32, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer or integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.sshl.sat.v4i32(<4 x i32>, <4 x i32>)
declare <4 x float> @llvm.sshl.sat.v4i32(<4 x i32>, <4 x i32>)

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector (overload type 0 is <16 x float>), but got ptr
; CHECK-NEXT: declare <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr>, ptr, <16 x float>)
declare <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr>, ptr, <16 x float>)

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with 8 elements (overload type 0 is <8 x float>), but got <16 x i1>
; CHECK-NEXT: declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, <16 x i1>, <8 x float>)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, <16 x i1>, <8 x float>)

; CHECK: intrinsic return type (overload type 0) expected any vector type, but got ptr
; CHECK-NEXT: declare ptr @llvm.masked.gather.p0.v8p0(<8 x ptr>, <8 x i1>, <8 x float>)
declare ptr @llvm.masked.gather.p0.v8p0(<8 x ptr>, <8 x i1>, <8 x float>)

; CHECK: intrinsic argument 0 type (vector of pointers to elements of overload type 0) expected vector (overload type 0 is <8 x float>), but got ptr
; CHECK-NEXT: declare <8 x float> @llvm.masked.gather.v8f32.p0(ptr, <8 x i1>, <8 x float>)
declare <8 x float> @llvm.masked.gather.v8f32.p0(ptr, <8 x i1>, <8 x float>)

; CHECK: intrinsic argument 0 type (vector of pointers to elements of overload type 0) expected vector of pointers with 8 elements (overload type 0 is <8 x float>), but got <8 x float>
; CHECK-NEXT: declare <8 x float> @llvm.masked.gather.v8f32.v8f32(<8 x float>, <8 x i1>, <8 x float>)
declare <8 x float> @llvm.masked.gather.v8f32.v8f32(<8 x float>, <8 x i1>, <8 x float>)

; CHECK: intrinsic argument 0 type (vector of pointers to elements of overload type 0) expected vector of pointers with 8 elements (overload type 0 is <8 x float>), but got <16 x ptr>
; CHECK-NEXT: declare <8 x float> @llvm.masked.gather.v8f32.v16p0(<16 x ptr>, <8 x i1>, <8 x float>)
declare <8 x float> @llvm.masked.gather.v8f32.v16p0(<16 x ptr>, <8 x i1>, <8 x float>)

; CHECK: intrinsic argument 2 type (matching overload type 0) expected <16 x i32>, but got <8 x i32>
; CHECK-NEXT: declare <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr>, <16 x i1>, <8 x i32>)
declare <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr>, <16 x i1>, <8 x i32>)

; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector (overload type 0 is <16 x float>), but got ptr
; CHECK-NEXT: declare void @llvm.masked.scatter.v16f32.v16p0(<16 x float>, <16 x ptr>, ptr)
declare void @llvm.masked.scatter.v16f32.v16p0(<16 x float>, <16 x ptr>, ptr)

; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector with 8 elements (overload type 0 is <8 x float>), but got <16 x i1>
; CHECK-NEXT: declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, <16 x i1>)
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, <16 x i1>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any vector type, but got ptr
; CHECK-NEXT: declare void @llvm.masked.scatter.p0.v8p0(ptr, <8 x ptr>, <8 x i1>)
declare void @llvm.masked.scatter.p0.v8p0(ptr, <8 x ptr>, <8 x i1>)

; CHECK: intrinsic argument 1 type (vector of pointers to elements of overload type 0) expected vector (overload type 0 is <8 x float>), but got ptr
; CHECK-NEXT: declare void @llvm.masked.scatter.v8f32.p0(<8 x float>, ptr, <8 x i1>)
declare void @llvm.masked.scatter.v8f32.p0(<8 x float>, ptr, <8 x i1>)

; CHECK: intrinsic argument 1 type (vector of pointers to elements of overload type 0) expected vector of pointers with 8 elements (overload type 0 is <8 x float>), but got <8 x float>
; CHECK-NEXT: declare void @llvm.masked.scatter.v8f32.v8f32(<8 x float>, <8 x float>, <8 x i1>)
declare void @llvm.masked.scatter.v8f32.v8f32(<8 x float>, <8 x float>, <8 x i1>)

; CHECK: intrinsic argument 1 type (vector of pointers to elements of overload type 0) expected vector of pointers with 8 elements (overload type 0 is <8 x float>), but got <16 x ptr>
; CHECK-NEXT: declare void @llvm.masked.scatter.v8f32.v16p0(<8 x float>, <16 x ptr>, <8 x i1>)
declare void @llvm.masked.scatter.v8f32.v16p0(<8 x float>, <16 x ptr>, <8 x i1>)

; CHECK: intrinsic was not defined with variable arguments!
; CHECK-NEXT: declare void @llvm.experimental.stackmap(i64, i32)
declare void @llvm.experimental.stackmap(i64, i32)

; CHECK: intrinsic was defined with variable arguments!
; CHECK-NEXT: declare void @llvm.donothing(...)
declare void @llvm.donothing(...)

; CHECK: intrinsic return struct element 0 type (matching overload type 0) expected <4 x i32>, but got <4 x i64>
; CHECK-NEXT: llvm.aarch64.neon.ld2.v4i32
declare { <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32(ptr %ptr)

; CHECK: intrinsic return struct element 1 type (matching overload type 0) expected <4 x i64>, but got <4 x i32>
; CHECK-NEXT: llvm.aarch64.neon.ld2lane.v4i64
declare { <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i64(<4 x i64>, <4 x i64>, i64, ptr)

; CHECK: intrinsic argument 0 type (matching overload type 0) expected <4 x i32>, but got <4 x i64>
; CHECK-NEXT: llvm.aarch64.neon.ld2lane.v4i32
declare { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i32(<4 x i64>, <4 x i32>, i64, ptr)

; CHECK: intrinsic return struct element 1 type (matching overload type 0) expected <4 x i32>, but got <4 x i64>
; CHECK-NEXT: llvm.aarch64.neon.ld3.v4i32
declare { <4 x i32>, <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32(ptr %ptr)

; CHECK: intrinsic return struct element 1 type (matching overload type 0) expected <4 x i64>, but got <4 x i32>
; CHECK-NEXT: llvm.aarch64.neon.ld3lane.v4i64
declare { <4 x i64>, <4 x i32>, <4 x i64> } @llvm.aarch64.neon.ld3lane.v4i64(<4 x i64>, <4 x i64>, <4 x i64>, i64, ptr)

; CHECK: intrinsic argument 0 type (matching overload type 0) expected <4 x i32>, but got <4 x i64>
; CHECK-NEXT: llvm.aarch64.neon.ld3lane.v4i32
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3lane.v4i32(<4 x i64>, <4 x i32>, <4 x i32>, i64, ptr)

; CHECK: intrinsic return struct element 2 type (matching overload type 0) expected <4 x i32>, but got <4 x i64>
; CHECK-NEXT: llvm.aarch64.neon.ld4.v4i32
declare { <4 x i32>, <4 x i32>, <4 x i64>, <4 x i32> } @llvm.aarch64.neon.ld4.v4i32(ptr %ptr)

; CHECK: intrinsic return struct element 2 type (matching overload type 0) expected <4 x i64>, but got <4 x i32>
; CHECK-NEXT: llvm.aarch64.neon.ld4lane.v4i64
declare { <4 x i64>, <4 x i64>, <4 x i32>, <4 x i64> } @llvm.aarch64.neon.ld4lane.v4i64(<4 x i64>, <4 x i64>, <4 x i64>, <4 x i64>, i64, ptr)

; CHECK: intrinsic argument 0 type (matching overload type 0) expected <4 x i32>, but got <4 x i64>
; CHECK-NEXT: llvm.aarch64.neon.ld4lane.v4i32
declare { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4lane.v4i32(<4 x i64>, <4 x i32>, <4 x i32>, <4 x i32>, i64, ptr)

; CHECK:      intrinsic return type (overload type 0) expected any integer vector, but got i32
; CHECK-NEXT: declare i32 @llvm.get.active.lane.mask.i32.i32(i32, i32)
declare i32 @llvm.get.active.lane.mask.i32.i32(i32, i32)

; CHECK:      intrinsic argument 0 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i1> @llvm.get.active.lane.mask.v4i1.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i1> @llvm.get.active.lane.mask.v4i1.v4i32(<4 x i32>, <4 x i32>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.experimental.get.vector.length.v4i32(<4 x i32>, i32, i1)
declare i32 @llvm.experimental.get.vector.length.v4i32(<4 x i32>, i32, i1)

; CHECK: intrinsic return type (overload type 0) expected any integer type, but got <2 x i32>
; CHECK-NEXT: declare <2 x i32> @llvm.get.dynamic.area.offset.v2i32()
declare <2 x i32> @llvm.get.dynamic.area.offset.v2i32()

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got i32
; CHECK-NEXT: declare i32 @llvm.stepvector.i32()
declare i32 @llvm.stepvector.i32()

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <vscale x 4 x float>
; CHECK-NEXT: declare <vscale x 4 x float> @llvm.stepvector.nxv4f32()
declare <vscale x 4 x float> @llvm.stepvector.nxv4f32()

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memcpy.element.unordered.atomic.p0.p0.v4i32(ptr, ptr, <4 x i32>, i32)
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.v4i32(ptr, ptr, <4 x i32>, i32)

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memmove.element.unordered.atomic.p0.p0.v4i32(ptr, ptr, <4 x i32>, i32)
declare void @llvm.memmove.element.unordered.atomic.p0.p0.v4i32(ptr, ptr, <4 x i32>, i32)

; CHECK: intrinsic argument 2 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.memset.element.unordered.atomic.p0.v4i32(ptr, i8, <4 x i32>, i32)
declare void @llvm.memset.element.unordered.atomic.p0.v4i32(ptr, i8, <4 x i32>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.add.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.add.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.mul.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.mul.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.and.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.and.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.or.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.or.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.xor.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.xor.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.smax.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.smax.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.smin.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.smin.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.umax.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.umax.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.umin.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.umin.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fadd.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fadd.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fmul.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fmul.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fmax.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fmax.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fmin.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fmin.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fmaximum.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fmaximum.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fminimum.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fminimum.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vp.sdiv.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)
declare <4 x float> @llvm.vp.sdiv.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vp.udiv.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)
declare <4 x float> @llvm.vp.udiv.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vp.srem.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)
declare <4 x float> @llvm.vp.srem.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vp.urem.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)
declare <4 x float> @llvm.vp.urem.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.vp.cttz.elts.v4i32.v4i32(<4 x i32>, i1, <4 x i1>, i32)
declare <4 x i32> @llvm.vp.cttz.elts.v4i32.v4i32(<4 x i32>, i1, <4 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got i32
; CHECK-NEXT: declare i32 @llvm.vp.cttz.elts.i32.i32(i32, i1, i1, i32)
declare i32 @llvm.vp.cttz.elts.i32.i32(i32, i1, i1, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare i32 @llvm.vp.cttz.elts.i32.v4f32(<4 x float>, i1, <4 x i1>, i32)
declare i32 @llvm.vp.cttz.elts.i32.v4f32(<4 x float>, i1, <4 x i1>, i32)

; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vp.strided.store.v4f32.p0.v4i32(<4 x float>, ptr, <4 x i32>, <4 x i1>, i32)
declare void @llvm.experimental.vp.strided.store.v4f32.p0.v4i32(<4 x float>, ptr, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x float> @llvm.experimental.vp.strided.load.v4f32.p0.v4i32(ptr, <4 x i32>, <4 x i1>, i32)
declare <4 x float> @llvm.experimental.vp.strided.load.v4f32.p0.v4i32(ptr, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any vector type, but got i32
; CHECK-NEXT: declare i32 @llvm.experimental.vp.splice.i32(i32, i32, i32, i1, i32, i32)
declare i32 @llvm.experimental.vp.splice.i32(i32, i32, i32, i1, i32, i32)

; CHECK: intrinsic argument 0 type (matching overload type 0) expected <4 x float>, but got <4 x i32>
; CHECK-NEXT: declare <4 x float> @llvm.matrix.transpose.v4f32.v4i32(<4 x i32>, i32, i32)
declare <4 x float> @llvm.matrix.transpose.v4f32.v4i32(<4 x i32>, i32, i32)

; CHECK: intrinsic argument 0 type (matching overload type 0) expected <4 x i32>, but got <4 x float>
; CHECK-NEXT: declare <4 x i32> @llvm.matrix.transpose.v4i32.v4f32(<4 x float>, i32, i32)
declare <4 x i32> @llvm.matrix.transpose.v4i32.v4f32(<4 x float>, i32, i32)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x float> @llvm.matrix.column.major.load.v4f32.v4i32(ptr, <4 x i32>, i1, i32, i32)
declare <4 x float> @llvm.matrix.column.major.load.v4f32.v4i32(ptr, <4 x i32>, i1, i32, i32)

; CHECK: intrinsic argument 2 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.matrix.column.major.store.v4f32.v4i32(<4 x float>, ptr, <4 x i32>, i1, i32, i32)
declare void @llvm.matrix.column.major.store.v4f32.v4i32(<4 x float>, ptr, <4 x i32>, i1, i32, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any vector type, but got float
; CHECK-NEXT: declare <4 x float> @llvm.matrix.multiply.v4f32.f32.f32(float, float, i32, i32, i32)
declare <4 x float> @llvm.matrix.multiply.v4f32.f32.f32(float, float, i32, i32, i32)

; CHECK: intrinsic return type (overload type 0) expected any vector type, but got float
; CHECK-NEXT: declare float @llvm.matrix.transpose.f32(float, i32, i32)
declare float @llvm.matrix.transpose.f32(float, i32, i32)

; CHECK: intrinsic return type (overload type 0) expected any vector type, but got float
; CHECK-NEXT: declare float @llvm.matrix.column.major.load.f32.i64(ptr, i64, i1, i32, i32)
declare float @llvm.matrix.column.major.load.f32.i64(ptr, i64, i1, i32, i32)

; CHECK: intrinsic argument 0 type (overload type 0) expected any vector type, but got float
; CHECK-NEXT: declare void @llvm.matrix.column.major.store.f32.i64(float, ptr, i64, i1, i32, i32)
declare void @llvm.matrix.column.major.store.f32.i64(float, ptr, i64, i1, i32, i32)

; CHECK: intrinsic argument 0 type (matching overload type 0) expected i32, but got i64
; CHECK-NEXT: declare i32 @llvm.callbr.landingpad.i64(i64)
declare i32 @llvm.callbr.landingpad.i64(i64)

; CHECK: intrinsic argument 0 type (overload type 1) expected any vector type, but got i32
; CHECK-NEXT: declare <4 x i32> @llvm.vector.extract.v4i32.i32(i32, i64)
declare <4 x i32> @llvm.vector.extract.v4i32.i32(i32, i64)

; CHECK: intrinsic argument 1 type (overload type 1) expected any vector type, but got i32
; CHECK-NEXT: declare <8 x i32> @llvm.vector.insert.v8i32.i32(<8 x i32>, i32, i64)
declare <8 x i32> @llvm.vector.insert.v8i32.i32(<8 x i32>, i32, i64)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vector.partial.reduce.add.v4f32.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.vector.partial.reduce.add.v4f32.v4f32(<4 x float>, <4 x float>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vector.partial.reduce.add.v4i32.v4f32(<4 x i32>, <4 x float>)
declare <4 x float> @llvm.vector.partial.reduce.add.v4i32.v4f32(<4 x i32>, <4 x float>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.vector.partial.reduce.fadd.v4i32.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.vector.partial.reduce.fadd.v4i32.v4i32(<4 x i32>, <4 x i32>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any fp vector, but got float
; CHECK-NEXT: declare float @llvm.vector.partial.reduce.fadd.f32.f32(float, float)
declare float @llvm.vector.partial.reduce.fadd.f32.f32(float, float)

; CHECK: intrinsic argument 1 type (overload type 1) expected any fp vector, but got float
; CHECK-NEXT: declare <4 x float> @llvm.vector.partial.reduce.fadd.v4f32.f32(<4 x float>, float)
declare <4 x float> @llvm.vector.partial.reduce.fadd.v4f32.f32(<4 x float>, float)

; CHECK: intrinsic return type expected void, but got float
; CHECK-NEXT: declare float @llvm.set.rounding(i32)
declare float @llvm.set.rounding(i32)

; CHECK: intrinsic return type expected x86_mmx (<1 x i64>), but got x86_amx
; CHECK-NEXT: declare x86_amx @llvm.x86.sse.cvtps2pi(<4 x float>)
declare x86_amx @llvm.x86.sse.cvtps2pi(<4 x float>)

; CHECK: intrinsic argument 1 type expected x86_mmx (<1 x i64>), but got i32
; CHECK-NEXT: declare <4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float>, i32)
declare <4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float>, i32)

; CHECK: intrinsic return type expected x86_amx, but got float
; CHECK-NEXT: declare float @llvm.x86.tileloadd64.internal(i16, i16, ptr, i64)
declare float @llvm.x86.tileloadd64.internal(i16, i16, ptr, i64)

; CHECK: intrinsic return type expected token, but got i32
; CHECK-NEXT: declare i32 @llvm.call.preallocated.setup(i32)
declare i32 @llvm.call.preallocated.setup(i32)

; CHECK: intrinsic argument 1 type expected metadata, but got i16
; CHECK-NEXT: declare double @llvm.fptrunc.round.f64.f64(double, i16)
declare double @llvm.fptrunc.round.f64.f64(double, i16)

; CHECK: intrinsic argument 0 type expected half, but got i16
; CHECK-NEXT: declare half @llvm.nvvm.fma.rn.f16(i16, half, half)
declare half @llvm.nvvm.fma.rn.f16(i16, half, half)

; CHECK: intrinsic return type expected bfloat, but got half
; CHECK-NEXT: declare half @llvm.arm.neon.vcvtbfp2bf(float)
declare half @llvm.arm.neon.vcvtbfp2bf(float)

; CHECK: intrinsic argument 2 type expected float, but got half
; CHECK-NEXT: declare float @llvm.x86.avx512.vfmadd.f32(float, float, half, i32)
declare float @llvm.x86.avx512.vfmadd.f32(float, float, half, i32)

; CHECK: intrinsic argument 2 type expected double, but got half
; CHECK-NEXT: declare i32 @llvm.expect.with.probability.i32(i32, i32, half)
declare i32 @llvm.expect.with.probability.i32(i32, i32, half)

; CHECK: intrinsic argument 0 type expected fp128, but got double
; CHECK-NEXT: declare double @llvm.ppc.truncf128.round.to.odd(double)
declare double @llvm.ppc.truncf128.round.to.odd(double)

; CHECK: intrinsic argument 0 type expected ppc_fp128, but got double
; CHECK-NEXT: declare double @llvm.ppc.unpack.longdouble(double, i32)
declare double @llvm.ppc.unpack.longdouble(double, i32)

; CHECK: intrinsic return type expected i64, but got double
; CHECK-NEXT: declare double @llvm.readcyclecounter()
declare double @llvm.readcyclecounter()

; CHECK: intrinsic argument 0 type expected aarch64.svcount, but got i32
; CHECK-NEXT: declare <4 x i32> @llvm.aarch64.sve.pext(i32, i32)
declare <4 x i32> @llvm.aarch64.sve.pext(i32, i32)

; CHECK: intrinsic argument 1 type expected ptr, but got i32
; CHECK-NEXT: declare void @llvm.gcroot(ptr, i32)
declare void @llvm.gcroot(ptr, i32)

; CHECK: intrinsic argument 0 type expected ptr addrspace(1), but got ptr
; CHECK-NEXT: declare void @llvm.nvvm.applypriority.global.L2.evict.normal(ptr, i64)
declare void @llvm.nvvm.applypriority.global.L2.evict.normal(ptr, i64)

; CHECK: intrinsic argument 0 type (extended overload type 0) expected i64 (overload type 0 is i32), but got i32
; CHECK-NEXT: declare i32 @llvm.aarch64.neon.sqxtn.i32(i32)
declare i32 @llvm.aarch64.neon.sqxtn.i32(i32)

; CHECK: intrinsic argument 0 type (extended overload type 0) expected <4 x i64> (overload type 0 is <4 x i32>), but got i64
; CHECK-NEXT: declare <4 x i32> @llvm.aarch64.neon.sqxtn.v4i32(i64)
declare <4 x i32> @llvm.aarch64.neon.sqxtn.v4i32(i64)

; CHECK: intrinsic argument 0 is truncated overload type 0, so overload type 0 expected int or vector of int, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.aarch64.neon.smull.v4f32(i64, i64)
declare <4 x float> @llvm.aarch64.neon.smull.v4f32(i64, i64)

; CHECK: intrinsic argument 1 type (truncated overload type 0) expected <4 x i32> (overload type 0 is <4 x i64>), but got i64
; CHECK-NEXT: declare <4 x i64> @llvm.aarch64.neon.smull.v4i64(<4 x i32>, i64)
declare <4 x i64> @llvm.aarch64.neon.smull.v4i64(<4 x i32>, i64)

; CHECK: intrinsic argument 0 is 1/nth (n=3) elements vector of overload type 0, so overload type 0 expected vector with multiple of 3 elements, but got <4 x i64>
; CHECK-NEXT: declare <4 x i64> @llvm.vector.interleave3.v4i64(i32, i32, i32)
declare <4 x i64> @llvm.vector.interleave3.v4i64(i32, i32, i32)

; CHECK: intrinsic argument 0 type (1/nth (n=3) elements vector of overload type 0) expected <4 x i64> (overload type 0 is <12 x i64>), but got <4 x i32>
; CHECK-NEXT: declare <12 x i64> @llvm.vector.interleave3.v12i64(<4 x i32>, <4 x i32>, i32)
declare <12 x i64> @llvm.vector.interleave3.v12i64(<4 x i32>, <4 x i32>, i32)

; CHECK: intrinsic return type (same vector width of overload type 0) expected vector (overload type 0 is <2 x float>), but got i1
; CHECK-NEXT: declare i1 @llvm.is.fpclass.v2f32(<2 x float>, i32)
declare i1 @llvm.is.fpclass.v2f32(<2 x float>, i32)

; CHECK: intrinsic return type (same vector width of overload type 0) expected scalar (overload type 0 is float), but got <2 x i1>
; CHECK-NEXT: declare <2 x i1> @llvm.is.fpclass.f32(float, i32)
declare <2 x i1> @llvm.is.fpclass.f32(float, i32)

; CHECK: intrinsic return type (same vector width of overload type 0) expected vector with 4 elements (overload type 0 is <4 x float>), but got <2 x i1>
; CHECK-NEXT: declare <2 x i1> @llvm.is.fpclass.v4f32(<4 x float>, i32)
declare <2 x i1> @llvm.is.fpclass.v4f32(<4 x float>, i32)

; CHECK: intrinsic argument 0 type (subdivided by 2 vector of overload type 0) expected <8 x i16> (overload type 0 is <4 x i32>), but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.aarch64.sve.sunpkhi(<4 x i32>)
declare <4 x i32> @llvm.aarch64.sve.sunpkhi(<4 x i32>)

; CHECK: intrinsic argument 2 type (subdivided by 4 vector of overload type 0) expected <16 x i8> (overload type 0 is <4 x i32>), but got <16 x i4>
; CHECK-NEXT: declare <4 x i32> @llvm.aarch64.sve.sdot.v4i32(<4 x i32>, <16 x i8>, <16 x i4>)
declare <4 x i32> @llvm.aarch64.sve.sdot.v4i32(<4 x i32>, <16 x i8>, <16 x i4>)

; CHECK: intrinsic argument 2 type (subdivided by 4 vector of overload type 0) expected <16 x half> (overload type 0 is <4 x double>), but got float
; CHECK-NEXT: declare <4 x double> @llvm.aarch64.sve.sdot.v4f64(<4 x double>, <16 x half>, float)
declare <4 x double> @llvm.aarch64.sve.sdot.v4f64(<4 x double>, <16 x half>, float)

; CHECK: intrinsic argument 2 type (vector of bitcasts to int of overload type 0) expected <4 x i32> (overload type 0 is <4 x float>), but got i32
; CHECK-NEXT: declare <4 x float> @llvm.riscv.vrgather.vv.v4f32.i32(<4 x float>, <4 x float>, i32, i32)
declare <4 x float> @llvm.riscv.vrgather.vv.v4f32.i32(<4 x float>, <4 x float>, i32, i32)

; CHECK: intrinsic has incorrect number of args. Expected 1, but got 2
; CHECK-NEXT: declare i64 @llvm.experimental.gc.get.pointer.offset.p0p0(ptr, ptr)
declare i64 @llvm.experimental.gc.get.pointer.offset.p0p0(ptr, ptr)

; CHECK: intrinsic return type expected i64, but got i32
; CHECK-NEXT: declare i32 @llvm.experimental.gc.get.pointer.offset.p0(ptr)
declare i32 @llvm.experimental.gc.get.pointer.offset.p0(ptr)

; CHECK: intrinsic argument 0 type (overload type 0) expected any pointer type, but got i32
; CHECK-NEXT: declare i64 @llvm.experimental.gc.get.pointer.offset.i32(i32)
declare i64 @llvm.experimental.gc.get.pointer.offset.i32(i32)

; CHECK: intrinsic has incorrect number of args. Expected 1, but got 2
; CHECK-NEXT: declare ptr @llvm.experimental.gc.get.pointer.base.p0p0(ptr, ptr)
declare ptr @llvm.experimental.gc.get.pointer.base.p0p0(ptr, ptr)

; CHECK: intrinsic return type (overload type 0) expected any pointer type, but got i32
; CHECK-NEXT: declare i32 @llvm.experimental.gc.get.pointer.base.i32.p0(ptr)
declare i32 @llvm.experimental.gc.get.pointer.base.i32.p0(ptr)

; CHECK: intrinsic argument 0 type (matching overload type 0) expected ptr, but got ptr addrspace(1)
; CHECK-NEXT: declare ptr @llvm.experimental.gc.get.pointer.base.p0.p1(ptr addrspace(1))
declare ptr @llvm.experimental.gc.get.pointer.base.p0.p1(ptr addrspace(1))

; CHECK: intrinsic has incorrect number of args. Expected 4, but got 3
; CHECK-NEXT: ; {{.*}}
; CHECK-NEXT: declare void @llvm.memset.i64(ptr captures(none), i8, i64)
declare void @llvm.memset.i64(ptr nocapture, i8, i64) nounwind

; CHECK: intrinsic has incorrect number of args. Expected 4, but got 3
; CHECK-NEXT: ; {{.*}}
; CHECK-NEXT: declare void @llvm.memcpy.i64(ptr captures(none), i8, i64)
declare void @llvm.memcpy.i64(ptr nocapture, i8, i64) nounwind

; CHECK: intrinsic has incorrect number of args. Expected 4, but got 3
; CHECK-NEXT: ; {{.*}}
; CHECK-NEXT: declare void @llvm.memmove.i64(ptr captures(none), i8, i64)
declare void @llvm.memmove.i64(ptr nocapture, i8, i64) nounwind

; CHECK: intrinsic return type (overload type 0) expected any manglable type, but got token
; CHECK-NEXT: declare token @llvm.ssa.copy.token(token)
declare token @llvm.ssa.copy.token(token)

; CHECK: intrinsic argument 0 type (overload type 0) expected any manglable type, but got token
; CHECK-NEXT: declare i1 @llvm.is.constant.token(token)
declare i1 @llvm.is.constant.token(token)

; CHECK: intrinsic return type (vector element of overload type 0) expected double (overload type 0 is <2 x double>), but got float
; CHECK-NEXT: declare float @llvm.vector.reduce.fadd.f32.f64.v2f64(double, <2 x double>)
declare float @llvm.vector.reduce.fadd.f32.f64.v2f64(double, <2 x double>)

; CHECK: intrinsic argument 0 type (vector element of overload type 0) expected double (overload type 0 is <2 x double>), but got float
; CHECK-NEXT: declare double @llvm.vector.reduce.fadd.f64.f32.v2f64(float, <2 x double>)
declare double @llvm.vector.reduce.fadd.f64.f32.v2f64(float, <2 x double>)

; CHECK: intrinsic return type (vector element of overload type 0) expected double (overload type 0 is <2 x double>), but got <2 x double>
; CHECK-NEXT: declare <2 x double> @llvm.vector.reduce.fadd.v2f64.f64.v2f64(double, <2 x double>)
declare <2 x double> @llvm.vector.reduce.fadd.v2f64.f64.v2f64(double, <2 x double>)

; CHECK: intrinsic argument 0 type (vector element of overload type 0) expected double (overload type 0 is <2 x double>), but got <2 x double>
; CHECK-NEXT: declare double @llvm.vector.reduce.fadd.f64.v2f64.v2f64(<2 x double>, <2 x double>)
declare double @llvm.vector.reduce.fadd.f64.v2f64.v2f64(<2 x double>, <2 x double>)

; CHECK: intrinsic return type expected bfloat, but got half
; CHECK-NEXT: declare half @llvm.nvvm.neg.bf16(bfloat)
declare half @llvm.nvvm.neg.bf16(bfloat)

; CHECK: intrinsic argument 1 type expected bfloat, but got half
; CHECK-NEXT: declare bfloat @llvm.nvvm.fmax.bf16(bfloat, half)
declare bfloat @llvm.nvvm.fmax.bf16(bfloat, half)

; CHECK: intrinsic has incorrect number of args. Expected 2, but got 1
; CHECK-NEXT: declare bfloat @llvm.nvvm.fmin.bf16(bfloat)
declare bfloat @llvm.nvvm.fmin.bf16(bfloat)

; CHECK: intrinsic argument 0 type (overload type 1) expected any fp or fp vector, but got ptr
; CHECK-NEXT: declare i8 @llvm.convert.to.arbitrary.fp.i8.ptr(ptr, metadata, metadata, i1)
declare i8 @llvm.convert.to.arbitrary.fp.i8.ptr(ptr, metadata, metadata, i1)

; CHECK: intrinsic return type (overload type 0) expected any fp or fp vector, but got ptr
; CHECK-NEXT: declare ptr @llvm.convert.from.arbitrary.fp.ptr.i8(i8, metadata)
declare ptr @llvm.convert.from.arbitrary.fp.ptr.i8(i8, metadata)

; CHECK: intrinsic argument 0 type (overload type 1) expected any fp or fp vector, but got i32
; CHECK-NEXT: declare i8 @llvm.convert.to.arbitrary.fp.i8.i32(i32, metadata, metadata, i1)
declare i8 @llvm.convert.to.arbitrary.fp.i8.i32(i32, metadata, metadata, i1)

; CHECK: intrinsic return type (overload type 0) expected any fp or fp vector, but got i32
; CHECK-NEXT: declare i32 @llvm.convert.from.arbitrary.fp.i32.i8(i8, metadata)
declare i32 @llvm.convert.from.arbitrary.fp.i32.i8(i8, metadata)

; CHECK: intrinsic argument 0 type (overload type 1) expected any fp or fp vector, but got <4 x ptr>
; CHECK-NEXT: declare <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4ptr(<4 x ptr>, metadata, metadata, i1)
declare <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4ptr(<4 x ptr>, metadata, metadata, i1)

; CHECK: intrinsic argument 0 type expected token, but got i32
; CHECK-NEXT: declare ptr @llvm.eh.exceptionpointer.p0(i32)
declare ptr @llvm.eh.exceptionpointer.p0(i32)

; CHECK: intrinsic argument 0 type expected vector with 4 elements, but got <vscale x 4 x i16>
; CHECK-NEXT: declare <4 x float> @llvm.arm.neon.vcvthf2fp(<vscale x 4 x i16>)
declare <4 x float> @llvm.arm.neon.vcvthf2fp(<vscale x 4 x i16>)

; CHECK: intrinsic return type expected vector with 4 elements, but got <vscale x 4 x i16>
; CHECK-NEXT: declare <vscale x 4 x i16> @llvm.arm.neon.vcvtfp2hf(<vscale x 4 x float>)
declare <vscale x 4 x i16> @llvm.arm.neon.vcvtfp2hf(<vscale x 4 x float>)
