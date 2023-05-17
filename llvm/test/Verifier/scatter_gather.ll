; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; Mask is not a vector
; CHECK: Intrinsic has incorrect argument type!
define <16 x float> @gather2(<16 x ptr> %ptrs, ptr %mask, <16 x float> %passthru) {
  %res = call <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr> %ptrs, i32 4, ptr %mask, <16 x float> %passthru)
  ret <16 x float> %res
}
declare <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr>, i32, ptr, <16 x float>)

; Mask length != return length
; CHECK: Intrinsic has incorrect argument type!
define <8 x float> @gather3(<8 x ptr> %ptrs, <16 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr> %ptrs, i32 4, <16 x i1> %mask, <8 x float> %passthru)
  ret <8 x float> %res
}
declare <8 x float> @llvm.masked.gather.v8f32.v8p0(<8 x ptr>, i32, <16 x i1>, <8 x float>)

; Return type is not a vector
; CHECK: Intrinsic has incorrect return type!
define ptr @gather4(<8 x ptr> %ptrs, <8 x i1> %mask, <8 x float> %passthru) {
  %res = call ptr @llvm.masked.gather.p0.v8p0(<8 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x float> %passthru)
  ret ptr %res
}
declare ptr @llvm.masked.gather.p0.v8p0(<8 x ptr>, i32, <8 x i1>, <8 x float>)

; Value type is not a vector
; CHECK: Intrinsic has incorrect argument type!
define <8 x float> @gather5(ptr %ptrs, <8 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float> @llvm.masked.gather.v8f32.p0(ptr %ptrs, i32 4, <8 x i1> %mask, <8 x float> %passthru)
  ret <8 x float> %res
}
declare <8 x float> @llvm.masked.gather.v8f32.p0(ptr, i32, <8 x i1>, <8 x float>)

; Value type is not a vector of pointers
; CHECK: Intrinsic has incorrect argument type!
define <8 x float> @gather6(<8 x float> %ptrs, <8 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float> @llvm.masked.gather.v8f32.v8f32(<8 x float> %ptrs, i32 4, <8 x i1> %mask, <8 x float> %passthru)
  ret <8 x float> %res
}
declare <8 x float> @llvm.masked.gather.v8f32.v8f32(<8 x float>, i32, <8 x i1>, <8 x float>)

; Value length!= vector of pointers length
; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.masked.gather.v8f32.v16p0
define <8 x float> @gather8(<16 x ptr> %ptrs, <8 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float> @llvm.masked.gather.v8f32.v16p0(<16 x ptr> %ptrs, i32 4, <8 x i1> %mask, <8 x float> %passthru)
  ret <8 x float> %res
}
declare <8 x float> @llvm.masked.gather.v8f32.v16p0(<16 x ptr>, i32, <8 x i1>, <8 x float>)

; Passthru type doesn't match return type
; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.masked.gather.v16i32.v16p0
define <16 x i32> @gather9(<16 x ptr> %ptrs, <16 x i1> %mask, <8 x i32> %passthru) {
  %res = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %ptrs, i32 4, <16 x i1> %mask, <8 x i32> %passthru)
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr>, i32, <16 x i1>, <8 x i32>)

; Mask is not a vector
; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.masked.scatter.v16f32.v16p0
define void @scatter2(<16 x float> %value, <16 x ptr> %ptrs, ptr %mask) {
  call void @llvm.masked.scatter.v16f32.v16p0(<16 x float> %value, <16 x ptr> %ptrs, i32 4, ptr %mask)
  ret void
}
declare void @llvm.masked.scatter.v16f32.v16p0(<16 x float>, <16 x ptr>, i32, ptr)

; Mask length != value length
; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.masked.scatter.v8f32.v8p0
define void @scatter3(<8 x float> %value, <8 x ptr> %ptrs, <16 x i1> %mask) {
  call void @llvm.masked.scatter.v8f32.v8p0(<8 x float> %value, <8 x ptr> %ptrs, i32 4, <16 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.v8f32.v8p0(<8 x float>, <8 x ptr>, i32, <16 x i1>)

; Value type is not a vector
; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.masked.scatter.p0.v8p0
define void @scatter4(ptr %value, <8 x ptr> %ptrs, <8 x i1> %mask) {
  call void @llvm.masked.scatter.p0.v8p0(ptr %value, <8 x ptr> %ptrs, i32 4, <8 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.p0.v8p0(ptr, <8 x ptr>, i32, <8 x i1>)

; ptrs is not a vector
; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.masked.scatter.v8f32.p0
define void @scatter5(<8 x float> %value, ptr %ptrs, <8 x i1> %mask) {
  call void @llvm.masked.scatter.v8f32.p0(<8 x float> %value, ptr %ptrs, i32 4, <8 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.v8f32.p0(<8 x float>, ptr, i32, <8 x i1>)

; Value type is not a vector of pointers
; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.masked.scatter.v8f32.v8f32
define void @scatter6(<8 x float> %value, <8 x float> %ptrs, <8 x i1> %mask) {
  call void @llvm.masked.scatter.v8f32.v8f32(<8 x float> %value, <8 x float> %ptrs, i32 4, <8 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.v8f32.v8f32(<8 x float>, <8 x float>, i32, <8 x i1>)

; Value length!= vector of pointers length
; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.masked.scatter.v8f32.v16p0
define void @scatter8(<8 x float> %value, <16 x ptr> %ptrs, <8 x i1> %mask) {
  call void @llvm.masked.scatter.v8f32.v16p0(<8 x float> %value, <16 x ptr> %ptrs, i32 4, <8 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.v8f32.v16p0(<8 x float>, <16 x ptr>, i32, <8 x i1>)

