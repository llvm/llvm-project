; RUN: llc -mtriple=arm-eabi -mattr=+v8.2a,+neon,-fullfp16 -float-abi=hard < %s | FileCheck -check-prefix=NOFP16 %s

declare void @f16_user(half)
declare half @f16_result()

declare void @v2f16_user(<2 x half>)
declare <2 x half> @v2f16_result()

declare void @v4f16_user(<4 x half>)
declare <4 x half> @v4f16_result()

declare void @v8f16_user(<8 x half>)
declare <8 x half> @v8f16_result()

define void @f16_arg(half %arg, ptr %ptr) #0 {
  %fpext = call float @llvm.experimental.constrained.fpext.f32.f16(half %arg, metadata !"fpexcept.strict")
  store float %fpext, ptr %ptr
  ret void
}

define void @v2f16_arg(<2 x half> %arg, ptr %ptr) #0 {
  %fpext = call <2 x float> @llvm.experimental.constrained.fpext.v2f32.v2f16(<2 x half> %arg, metadata !"fpexcept.strict")
  store <2 x float> %fpext, ptr %ptr
  ret void
}

define void @v3f16_arg(<3 x half> %arg, ptr %ptr) #0 {
  %fpext = call <3 x float> @llvm.experimental.constrained.fpext.v3f32.v3f16(<3 x half> %arg, metadata !"fpexcept.strict")
  store <3 x float> %fpext, ptr %ptr
  ret void
}

define void @v4f16_arg(<4 x half> %arg, ptr %ptr) #0 {
  %fpext = call <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half> %arg, metadata !"fpexcept.strict")
  store <4 x float> %fpext, ptr %ptr
  ret void
}

define half @f16_return(float %arg) #0 {
  %fptrunc = call half @llvm.experimental.constrained.fptrunc.f16.f32(float %arg, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret half %fptrunc
}

define <2 x half> @v2f16_return(<2 x float> %arg) #0 {
  %fptrunc = call <2 x half> @llvm.experimental.constrained.fptrunc.v2f16.v2f32(<2 x float> %arg, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret <2 x half> %fptrunc
}

define <3 x half> @v3f16_return(<3 x float> %arg) #0 {
  %fptrunc = call <3 x half> @llvm.experimental.constrained.fptrunc.v3f16.v3f32(<3 x float> %arg, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret <3 x half> %fptrunc
}

define <4 x half> @v4f16_return(<4 x float> %arg) #0 {
  %fptrunc = call <4 x half> @llvm.experimental.constrained.fptrunc.v4f16.v4f32(<4 x float> %arg, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret <4 x half> %fptrunc
}

define void @outgoing_f16_arg(ptr %ptr) #0 {
  %val = load half, ptr %ptr
  call void @f16_user(half %val)
  ret void
}

define void @outgoing_v2f16_arg(ptr %ptr) #0 {
  %val = load <2 x half>, ptr %ptr
  call void @v2f16_user(<2 x half> %val)
  ret void
}

define void @outgoing_f16_return(ptr %ptr) #0 {
  %val = call half @f16_result()
  store half %val, ptr %ptr
  ret void
}

define void @outgoing_v2f16_return(ptr %ptr) #0 {
  %val = call <2 x half> @v2f16_result()
  store <2 x half> %val, ptr %ptr
  ret void
}

define void @outgoing_v4f16_return(ptr %ptr) #0 {
  %val = call <4 x half> @v4f16_result()
  store <4 x half> %val, ptr %ptr
  ret void
}

define void @outgoing_v8f16_return(ptr %ptr) #0 {
  %val = call <8 x half> @v8f16_result()
  store <8 x half> %val, ptr %ptr
  ret void
}

define half @call_split_type_used_outside_block_v8f16() #0 {
bb0:
  %split.ret.type = call <8 x half> @v8f16_result()
  br label %bb1

bb1:
  %extract = extractelement <8 x half> %split.ret.type, i32 0
  ret half %extract
}

declare float @llvm.experimental.constrained.fpext.f32.f16(half, metadata) #0
declare <2 x float> @llvm.experimental.constrained.fpext.v2f32.v2f16(<2 x half>, metadata) #0
declare <3 x float> @llvm.experimental.constrained.fpext.v3f32.v3f16(<3 x half>, metadata) #0
declare <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half>, metadata) #0

declare half @llvm.experimental.constrained.fptrunc.f16.f32(float, metadata, metadata) #0
declare <2 x half> @llvm.experimental.constrained.fptrunc.v2f16.v2f32(<2 x float>, metadata, metadata) #0
declare <3 x half> @llvm.experimental.constrained.fptrunc.v3f16.v3f32(<3 x float>, metadata, metadata) #0
declare <4 x half> @llvm.experimental.constrained.fptrunc.v4f16.v4f32(<4 x float>, metadata, metadata) #0

attributes #0 = { strictfp }
