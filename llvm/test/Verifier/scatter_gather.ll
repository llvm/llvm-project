; RUN: not opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s

; Mask is not a vector
; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector (overload type 0 is <16 x float>), but got ptr
; CHECK-NEXT: declare <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr>, ptr, <16 x float>)
declare <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr>, ptr, <16 x float>)

; Mask length != return length
; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with 8 elements (overload type 0 is <8 x float>), but got <16 x i1>
; CHECK-NEXT: declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, <16 x i1>, <8 x float>)
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, <16 x i1>, <8 x float>)

; Return type is not a vector
; CHECK: intrinsic return type (overload type 0) expected any vector type, but got ptr
; CHECK-NEXT: declare ptr @llvm.masked.gather.p0.v8p0(<8 x ptr>, <8 x i1>, <8 x float>)
declare ptr @llvm.masked.gather.p0.v8p0(<8 x ptr>, <8 x i1>, <8 x float>)

; Value type is not a vector
; CHECK: intrinsic argument 0 type (vector of pointers to elements of overload type 0) expected vector (overload type 0 is <8 x float>), but got ptr
; CHECK-NEXT: declare <8 x float> @llvm.masked.gather.v8f32.p0(ptr, <8 x i1>, <8 x float>)
declare <8 x float> @llvm.masked.gather.v8f32.p0(ptr, <8 x i1>, <8 x float>)

; Value type is not a vector of pointers
; CHECK: intrinsic argument 0 type (vector of pointers to elements of overload type 0) expected vector of pointers with 8 elements (overload type 0 is <8 x float>), but got <8 x float>
; CHECK-NEXT: declare <8 x float> @llvm.masked.gather.v8f32.v8f32(<8 x float>, <8 x i1>, <8 x float>)
declare <8 x float> @llvm.masked.gather.v8f32.v8f32(<8 x float>, <8 x i1>, <8 x float>)

; Value length!= vector of pointers length
; CHECK: intrinsic argument 0 type (vector of pointers to elements of overload type 0) expected vector of pointers with 8 elements (overload type 0 is <8 x float>), but got <16 x ptr>
; CHECK-NEXT: declare <8 x float> @llvm.masked.gather.v8f32.v16p0(<16 x ptr>, <8 x i1>, <8 x float>)
declare <8 x float> @llvm.masked.gather.v8f32.v16p0(<16 x ptr>, <8 x i1>, <8 x float>)

; Passthru type doesn't match return type
; CHECK: intrinsic argument 2 type (matching overload type 0) expected <16 x i32>, but got <8 x i32>
; CHECK-NEXT: declare <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr>, <16 x i1>, <8 x i32>)
declare <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr>, <16 x i1>, <8 x i32>)

; Mask is not a vector
; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector (overload type 0 is <16 x float>), but got ptr
; CHECK-NEXT: declare void @llvm.masked.scatter.v16f32.v16p0(<16 x float>, <16 x ptr>, ptr)
declare void @llvm.masked.scatter.v16f32.v16p0(<16 x float>, <16 x ptr>, ptr)

; Mask length != value length
; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector with 8 elements (overload type 0 is <8 x float>), but got <16 x i1>
; CHECK-NEXT: declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, <16 x i1>)
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, <16 x i1>)

; Value type is not a vector
; CHECK: intrinsic argument 0 type (overload type 0) expected any vector type, but got ptr
; CHECK-NEXT: declare void @llvm.masked.scatter.p0.v8p0(ptr, <8 x ptr>, <8 x i1>)
declare void @llvm.masked.scatter.p0.v8p0(ptr, <8 x ptr>, <8 x i1>)

; ptrs is not a vector
; CHECK: intrinsic argument 1 type (vector of pointers to elements of overload type 0) expected vector (overload type 0 is <8 x float>), but got ptr
; CHECK-NEXT: declare void @llvm.masked.scatter.v8f32.p0(<8 x float>, ptr, <8 x i1>)
declare void @llvm.masked.scatter.v8f32.p0(<8 x float>, ptr, <8 x i1>)

; Value type is not a vector of pointers
; CHECK: intrinsic argument 1 type (vector of pointers to elements of overload type 0) expected vector of pointers with 8 elements (overload type 0 is <8 x float>), but got <8 x float>
; CHECK-NEXT: declare void @llvm.masked.scatter.v8f32.v8f32(<8 x float>, <8 x float>, <8 x i1>)
declare void @llvm.masked.scatter.v8f32.v8f32(<8 x float>, <8 x float>, <8 x i1>)

; Value length!= vector of pointers length
; CHECK: intrinsic argument 1 type (vector of pointers to elements of overload type 0) expected vector of pointers with 8 elements (overload type 0 is <8 x float>), but got <16 x ptr>
; CHECK-NEXT: declare void @llvm.masked.scatter.v8f32.v16p0(<8 x float>, <16 x ptr>, <8 x i1>)
declare void @llvm.masked.scatter.v8f32.v16p0(<8 x float>, <16 x ptr>, <8 x i1>)

