; RUN: split-file %s %t

;--- test0.ll
; RUN: not opt -S -passes=verify -disable-output 2>&1 < %t/test0.ll | FileCheck %t/test0.ll

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
; CHECK-NEXT: declare half @llvm.nvvm.mul.rn.sat.f16(i16, half)
declare half @llvm.nvvm.mul.rn.sat.f16(i16, half)

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

; CHECK: intrinsic return type expected vector with 16 elements, but got i32
; CHECK-NEXT: declare i32 @llvm.aarch64.neon.pmull64(i64, i64)

; Correct return type is <16 x i8>
declare i32 @llvm.aarch64.neon.pmull64(i64, i64)

; CHECK: intrinsic argument 1 type expected ptr, but got i32
; CHECK-NEXT: declare void @llvm.gcroot(ptr, i32)
declare void @llvm.gcroot(ptr, i32)

; CHECK: intrinsic argument 0 type expected ptr addrspace(1), but got ptr
; CHECK-NEXT: declare void @llvm.nvvm.applypriority.global.L2.evict.normal(ptr, i64)
declare void @llvm.nvvm.applypriority.global.L2.evict.normal(ptr, i64)

; CHECK: intrinsic return type expected literal non-packed struct with 2 elements, but got void
; CHECK-NEXT: declare void @llvm.nvvm.elect.sync(i32)

; Expected return type is { i32, i1 }.
declare void @llvm.nvvm.elect.sync(i32)

; CHECK: intrinsic argument 0 type (extended overload type 0) expected i64 (overload type 0 is i32), but got i32
; CHECK-NEXT: declare i32 @llvm.aarch64.neon.sqxtn.i32(i32)

; Type signature = [llvm_anyint_ty], [LLVMExtendedType<0>]
declare i32 @llvm.aarch64.neon.sqxtn.i32(i32)

; CHECK: intrinsic argument 0 type (extended overload type 0) expected <4 x i64> (overload type 0 is <4 x i32>), but got i64
; CHECK-NEXT: declare <4 x i32> @llvm.aarch64.neon.sqxtn.v4i32(i64)

; Type signature = [llvm_anyint_ty], [LLVMExtendedType<0>]
declare <4 x i32> @llvm.aarch64.neon.sqxtn.v4i32(i64)

; CHECK: intrinsic argument 0 is truncated overload type 0, so overload type 0 expected int or vector of int, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.aarch64.neon.smull.v4f32(i64, i64)

; Type signature = [llvm_anyvector_ty], [LLVMTruncatedType<0>, LLVMTruncatedType<0>]
declare <4 x float> @llvm.aarch64.neon.smull.v4f32(i64, i64)

; CHECK: intrinsic argument 1 type (truncated overload type 0) expected <4 x i32> (overload type 0 is <4 x i64>), but got i64
; CHECK-NEXT: declare <4 x i64> @llvm.aarch64.neon.smull.v4i64(<4 x i32>, i64)

; Type signature = [llvm_anyvector_ty], [LLVMTruncatedType<0>, LLVMTruncatedType<0>]
declare <4 x i64> @llvm.aarch64.neon.smull.v4i64(<4 x i32>, i64)

; CHECK: intrinsic argument 0 is 1/nth (n=3) elements vector of overload type 0, so overload type 0 expected vector with multiple of 3 elements, but got <4 x i64>
; CHECK-NEXT: declare <4 x i64> @llvm.vector.interleave3.v4i64(i32, i32, i32)

; Type signature = [llvm_anyvector_ty], !listsplat(LLVMOneNthElementsVectorType<0, n>, n)
declare <4 x i64> @llvm.vector.interleave3.v4i64(i32, i32, i32)

; CHECK: intrinsic argument 0 type (1/nth (n=3) elements vector of overload type 0) expected <4 x i64> (overload type 0 is <12 x i64>), but got <4 x i32>
; CHECK-NEXT: declare <12 x i64> @llvm.vector.interleave3.v12i64(<4 x i32>, <4 x i32>, i32)

; Type signature = [llvm_anyvector_ty], !listsplat(LLVMOneNthElementsVectorType<0, n>, n)
declare <12 x i64> @llvm.vector.interleave3.v12i64(<4 x i32>, <4 x i32>, i32)

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with vscale x 4 elements (overload type 0 is <vscale x 4 x i32>), but got <4 x i1>
; CHECK-NEXT: declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr, <4 x i1>, <vscale x 4 x i32>)

declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr, <4 x i1>, <vscale x 4 x i32>)

; CHECK: intrinsic return type (same vector width of overload type 0) expected vector (overload type 0 is <2 x float>), but got i1
; CHECK-NEXT: declare i1 @llvm.is.fpclass.v2f32(<2 x float>, i32)

; type signature = [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>], [llvm_anyfloat_ty, llvm_i32_ty]
declare i1 @llvm.is.fpclass.v2f32(<2 x float>, i32)

; CHECK: intrinsic return type (same vector width of overload type 0) expected scalar (overload type 0 is float), but got <2 x i1>
; CHECK-NEXT: declare <2 x i1> @llvm.is.fpclass.f32(float, i32)

; type signature = [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>], [llvm_anyfloat_ty, llvm_i32_ty]
declare <2 x i1> @llvm.is.fpclass.f32(float, i32)

; CHECK: intrinsic return type (same vector width of overload type 0) expected vector with 4 elements (overload type 0 is <4 x float>), but got <2 x i1>
; CHECK-NEXT: declare <2 x i1> @llvm.is.fpclass.v4f32(<4 x float>, i32)

; type signature = [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>], [llvm_anyfloat_ty, llvm_i32_ty],
declare <2 x i1> @llvm.is.fpclass.v4f32(<4 x float>, i32)

; CHECK: intrinsic argument 0 type (subdivided by 2 vector of overload type 0) expected <8 x i16> (overload type 0 is <4 x i32>), but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.aarch64.sve.sunpkhi(<4 x i32>)

; type signature = [llvm_anyvector_ty], [LLVMSubdivide2VectorType<0>]
declare <4 x i32> @llvm.aarch64.sve.sunpkhi(<4 x i32>)

; CHECK: intrinsic argument 2 type (subdivided by 4 vector of overload type 0) expected <16 x i8> (overload type 0 is <4 x i32>), but got <16 x i4>
; CHECK-NEXT: declare <4 x i32> @llvm.aarch64.sve.sdot.v4i32(<4 x i32>, <16 x i8>, <16 x i4>)

; type signature = [llvm_anyvector_ty], [LLVMMatchType<0>, LLVMSubdivide4VectorType<0>, LLVMSubdivide4VectorType<0>]
declare <4 x i32> @llvm.aarch64.sve.sdot.v4i32(<4 x i32>, <16 x i8>, <16 x i4>)

; CHECK: intrinsic argument 2 type (subdivided by 4 vector of overload type 0) expected <16 x half> (overload type 0 is <4 x double>), but got float
; CHECK-NEXT: declare <4 x double> @llvm.aarch64.sve.sdot.v4f64(<4 x double>, <16 x half>, float)

; type signature = [llvm_anyvector_ty], [LLVMMatchType<0>, LLVMSubdivide4VectorType<0>, LLVMSubdivide4VectorType<0>]
declare <4 x double> @llvm.aarch64.sve.sdot.v4f64(<4 x double>, <16 x half>, float)

; CHECK: intrinsic argument 2 type (vector of bitcasts to int of overload type 0) expected <4 x i32> (overload type 0 is <4 x float>), but got i32
; CHECK-NEXT: declare <4 x float> @llvm.riscv.vrgather.vv.v4f32.i32(<4 x float>, <4 x float>, i32, i32)

; type signature = [llvm_anyvector_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMVectorOfBitcastsToInt<0>, llvm_anyint_ty]
declare <4 x float> @llvm.riscv.vrgather.vv.v4f32.i32(<4 x float>, <4 x float>, i32, i32)

; CHECK: intrinsic has incorrect number of args. Expected 1, but got 2
; CHECK-NEXT: declare i64 @llvm.experimental.gc.get.pointer.offset.p0p0(ptr, ptr)
declare i64 @llvm.experimental.gc.get.pointer.offset.p0p0(ptr, ptr)

; CHECK: intrinsic return type expected i64, but got i32
; CHECK-NEXT: declare i32 @llvm.experimental.gc.get.pointer.offset.p0(ptr)
declare i32 @llvm.experimental.gc.get.pointer.offset.p0(ptr)

; CHECK:intrinsic argument 0 type (overload type 0) expected any pointer type, but got i32
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

;--- test1.ll
; RUN: not opt -S -passes=verify -disable-output 2>&1 < %t/test1.ll | FileCheck %t/test1.ll

; CHECK: intrinsic return vector element type expected i8, but got i32
; CHECK-NEXT: declare <16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)
declare <16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)

; CHECK: intrinsic return struct element 1 type expected i1, but got i2
; CHECK-NEXT: declare { i32, i2 } @llvm.nvvm.elect.sync(i32)

; Expected return type is { i32, i1 }.
declare { i32, i2 } @llvm.nvvm.elect.sync(i32)


;--- test2.ll
; RUN: not opt -S -passes=verify -disable-output 2>&1 < %t/test2.ll | FileCheck %t/test2.ll

; CHECK: intrinsic return type expected vector with 16 elements, but got <vscale x 16 x i32>
; CHECK-NEXT: declare <vscale x 16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)
declare <vscale x 16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)

; CHECK: intrinsic return type expected literal non-packed struct with 2 elements, but got { i32, i1, i1 }
; CHECK-NEXT: declare { i32, i1, i1 } @llvm.nvvm.elect.sync(i32)

; Expected return type is { i32, i1 }.
declare { i32, i1, i1 } @llvm.nvvm.elect.sync(i32)
