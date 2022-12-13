; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

; Mask is not a vector
; CHECK: Intrinsic has incorrect argument type!
define <16 x float> @gather2(<16 x float*> %ptrs, <16 x i1>* %mask, <16 x float> %passthru) {
  %res = call <16 x float> @llvm.masked.gather.v16f32.v16p0f32(<16 x float*> %ptrs, i32 4, <16 x i1>* %mask, <16 x float> %passthru)
  ret <16 x float> %res
}
declare <16 x float> @llvm.masked.gather.v16f32.v16p0f32(<16 x float*>, i32, <16 x i1>*, <16 x float>)

; Mask length != return length
; CHECK: Intrinsic has incorrect argument type!
define <8 x float> @gather3(<8 x float*> %ptrs, <16 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %ptrs, i32 4, <16 x i1> %mask, <8 x float> %passthru)
  ret <8 x float> %res
}
declare <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*>, i32, <16 x i1>, <8 x float>)

; Return type is not a vector
; CHECK: Intrinsic has incorrect return type!
define <8 x float>* @gather4(<8 x float*> %ptrs, <8 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float>* @llvm.masked.gather.p0v8f32.v8p0f32(<8 x float*> %ptrs, i32 4, <8 x i1> %mask, <8 x float> %passthru)
  ret <8 x float>* %res
}
declare <8 x float>* @llvm.masked.gather.p0v8f32.v8p0f32(<8 x float*>, i32, <8 x i1>, <8 x float>)

; Value type is not a vector
; CHECK: Intrinsic has incorrect argument type!
define <8 x float> @gather5(<8 x float*>* %ptrs, <8 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float> @llvm.masked.gather.v8f32.p0v8p0f32(<8 x float*>* %ptrs, i32 4, <8 x i1> %mask, <8 x float> %passthru)
  ret <8 x float> %res
}
declare <8 x float> @llvm.masked.gather.v8f32.p0v8p0f32(<8 x float*>*, i32, <8 x i1>, <8 x float>)

; Value type is not a vector of pointers
; CHECK: Intrinsic has incorrect argument type!
define <8 x float> @gather6(<8 x float> %ptrs, <8 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float> @llvm.masked.gather.v8f32.v8f32(<8 x float> %ptrs, i32 4, <8 x i1> %mask, <8 x float> %passthru)
  ret <8 x float> %res
}
declare <8 x float> @llvm.masked.gather.v8f32.v8f32(<8 x float>, i32, <8 x i1>, <8 x float>)

; Value element type != vector of pointers element
; CHECK: Intrinsic has incorrect argument type!
define <8 x float> @gather7(<8 x double*> %ptrs, <8 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float> @llvm.masked.gather.v8f32.v8p0f64(<8 x double*> %ptrs, i32 4, <8 x i1> %mask, <8 x float> %passthru)
  ret <8 x float> %res
}
declare <8 x float> @llvm.masked.gather.v8f32.v8p0f64(<8 x double*>, i32, <8 x i1>, <8 x float>)

; Value length!= vector of pointers length
; CHECK: Intrinsic has incorrect argument type!
define <8 x float> @gather8(<16 x float*> %ptrs, <8 x i1> %mask, <8 x float> %passthru) {
  %res = call <8 x float> @llvm.masked.gather.v8f32.v16p0f32(<16 x float*> %ptrs, i32 4, <8 x i1> %mask, <8 x float> %passthru)
  ret <8 x float> %res
}
declare <8 x float> @llvm.masked.gather.v8f32.v16p0f32(<16 x float*>, i32, <8 x i1>, <8 x float>)

; Passthru type doesn't match return type
; CHECK: Intrinsic has incorrect argument type!
define <16 x i32> @gather9(<16 x i32*> %ptrs, <16 x i1> %mask, <8 x i32> %passthru) {
  %res = call <16 x i32> @llvm.masked.gather.v16i32.v16p0i32(<16 x i32*> %ptrs, i32 4, <16 x i1> %mask, <8 x i32> %passthru)
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.masked.gather.v16i32.v16p0i32(<16 x i32*>, i32, <16 x i1>, <8 x i32>)

; Mask is not a vector
; CHECK: Intrinsic has incorrect argument type!
define void @scatter2(<16 x float> %value, <16 x float*> %ptrs, <16 x i1>* %mask) {
  call void @llvm.masked.scatter.v16f32.v16p0f32(<16 x float> %value, <16 x float*> %ptrs, i32 4, <16 x i1>* %mask)
  ret void
}
declare void @llvm.masked.scatter.v16f32.v16p0f32(<16 x float>, <16 x float*>, i32, <16 x i1>*)

; Mask length != value length
; CHECK: Intrinsic has incorrect argument type!
define void @scatter3(<8 x float> %value, <8 x float*> %ptrs, <16 x i1> %mask) {
  call void @llvm.masked.scatter.v8f32.v8p0f32(<8 x float> %value, <8 x float*> %ptrs, i32 4, <16 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.v8f32.v8p0f32(<8 x float>, <8 x float*>, i32, <16 x i1>)

; Value type is not a vector
; CHECK: Intrinsic has incorrect argument type!
define void @scatter4(<8 x float>* %value, <8 x float*> %ptrs, <8 x i1> %mask) {
  call void @llvm.masked.scatter.p0v8f32.v8p0f32(<8 x float>* %value, <8 x float*> %ptrs, i32 4, <8 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.p0v8f32.v8p0f32(<8 x float>*, <8 x float*>, i32, <8 x i1>)

; ptrs is not a vector
; CHECK: Intrinsic has incorrect argument type!
define void @scatter5(<8 x float> %value, <8 x float*>* %ptrs, <8 x i1> %mask) {
  call void @llvm.masked.scatter.v8f32.p0v8p0f32(<8 x float> %value, <8 x float*>* %ptrs, i32 4, <8 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.v8f32.p0v8p0f32(<8 x float>, <8 x float*>*, i32, <8 x i1>)

; Value type is not a vector of pointers
; CHECK: Intrinsic has incorrect argument type!
define void @scatter6(<8 x float> %value, <8 x float> %ptrs, <8 x i1> %mask) {
  call void @llvm.masked.scatter.v8f32.v8f32(<8 x float> %value, <8 x float> %ptrs, i32 4, <8 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.v8f32.v8f32(<8 x float>, <8 x float>, i32, <8 x i1>)

; Value element type != vector of pointers element
; CHECK: Intrinsic has incorrect argument type!
define void @scatter7(<8 x float> %value, <8 x double*> %ptrs, <8 x i1> %mask) {
  call void @llvm.masked.scatter.v8f32.v8p0f64(<8 x float> %value, <8 x double*> %ptrs, i32 4, <8 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.v8f32.v8p0f64(<8 x float>, <8 x double*>, i32, <8 x i1>)

; Value length!= vector of pointers length
; CHECK: Intrinsic has incorrect argument type!
define void @scatter8(<8 x float> %value, <16 x float*> %ptrs, <8 x i1> %mask) {
  call void @llvm.masked.scatter.v8f32.v16p0f32(<8 x float> %value, <16 x float*> %ptrs, i32 4, <8 x i1> %mask)
  ret void
}
declare void @llvm.masked.scatter.v8f32.v16p0f32(<8 x float>, <16 x float*>, i32, <8 x i1>)

