; RUN: opt -S -dxil-resource-access -dxil-op-lower %s | FileCheck %s

; Verify atomicrmw through a dx.resource.getpointer of a texture is lowered to
; dx.op.atomicBinOp, with one coordinate operand per texture dimension.

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK-LABEL: define i32 @atomic_texture1d(
define i32 @atomic_texture1d(i32 %coord, i32 %value) {
  %texture = call target("dx.Texture", i32, 1, 0, 0, 1)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", i32, 1, 0, 0, 1) %texture, i32 %coord)

  ; CHECK: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 0, i32 %coord, i32 poison, i32 poison, i32 %value)
  %add = atomicrmw add ptr %ptr, i32 %value monotonic
  ret i32 %add
}

; CHECK-LABEL: define i32 @atomic_texture2d(
define i32 @atomic_texture2d(<2 x i32> %coords, i32 %value) {
  %texture = call target("dx.Texture", i32, 1, 0, 0, 2)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", i32, 1, 0, 0, 2) %texture, <2 x i32> %coords)

  ; CHECK: %[[X:.*]] = extractelement <2 x i32> %coords, i64 0
  ; CHECK: %[[Y:.*]] = extractelement <2 x i32> %coords, i64 1
  ; CHECK: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 6, i32 %[[X]], i32 %[[Y]], i32 poison, i32 %value)
  %umin = atomicrmw umin ptr %ptr, i32 %value monotonic
  ret i32 %umin
}

; CHECK-LABEL: define i32 @atomic_texture2darray(
define i32 @atomic_texture2darray(<3 x i32> %coords, i32 %value) {
  %texture = call target("dx.Texture", i32, 1, 0, 0, 7)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", i32, 1, 0, 0, 7) %texture, <3 x i32> %coords)

  ; CHECK: %[[X:.*]] = extractelement <3 x i32> %coords, i64 0
  ; CHECK: %[[Y:.*]] = extractelement <3 x i32> %coords, i64 1
  ; CHECK: %[[Z:.*]] = extractelement <3 x i32> %coords, i64 2
  ; CHECK: call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %{{.*}}, i32 8, i32 %[[X]], i32 %[[Y]], i32 %[[Z]], i32 %value)
  %xchg = atomicrmw xchg ptr %ptr, i32 %value monotonic
  ret i32 %xchg
}

; CHECK-LABEL: define i64 @atomic_texture3d_i64(
define i64 @atomic_texture3d_i64(<3 x i32> %coords, i64 %value) {
  %texture = call target("dx.Texture", i64, 1, 0, 0, 4)
      @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr null)
  %ptr = call ptr @llvm.dx.resource.getpointer(
      target("dx.Texture", i64, 1, 0, 0, 4) %texture, <3 x i32> %coords)

  ; CHECK: %[[X:.*]] = extractelement <3 x i32> %coords, i64 0
  ; CHECK: %[[Y:.*]] = extractelement <3 x i32> %coords, i64 1
  ; CHECK: %[[Z:.*]] = extractelement <3 x i32> %coords, i64 2
  ; CHECK: call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle %{{.*}}, i32 5, i32 %[[X]], i32 %[[Y]], i32 %[[Z]], i64 %value)
  %max = atomicrmw max ptr %ptr, i64 %value monotonic
  ret i64 %max
}
