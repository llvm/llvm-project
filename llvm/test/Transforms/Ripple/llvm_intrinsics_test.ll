; RUN: opt -passes=ripple -S %s | FileCheck %s
; ModuleID = 'llvm_intrinsics_test.ll'
source_filename = "llvm_intrinsics_test.ll"

define dso_local void @test_abs(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val = load i32, ptr %in_ptr, align 4
  ; CHECK: %[[RESULT:.*]] = call <32 x i32> @llvm.abs.v32i32
  %result = call i32 @llvm.abs.i32(i32 %val, i1 false)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @convert_i16_fp16_to_fp32(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr = getelementptr inbounds i16, ptr %in_array, i64 %idx
  %val = load i16, ptr %in_ptr, align 2
  ; CHECK: .ripple.call.loop.body.call.block:
  ; CHECK: %[[RESULT:.*]] = extractelement <64 x i16> %.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: call float @llvm.convert.from.fp16.f32(i16 %[[RESULT]])
  %result = call float @llvm.convert.from.fp16.f32(i16 %val)
  %out_ptr = getelementptr inbounds float, ptr %dest, i64 %idx
  store float %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_ctlz(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val = load i32, ptr %in_ptr, align 4
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 false)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %ctlz, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_cttz(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val = load i32, ptr %in_ptr, align 4
  ; CHECK: %[[RESULT:.*]] = call{{.*}}<32 x i32> @llvm.cttz.v32i32
  %cttz = call i32 @llvm.cttz.i32(i32 %val, i1 false)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %cttz, ptr %out_ptr, align 4
  ret void
}

; Current RipplePass does not handle metadata type
; define dso_local void @truncate_with_rounding(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
; entry:
;   %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
;   %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
;   %in_ptr = getelementptr inbounds double, ptr %in_array, i64 %idx
;   %val = load double, ptr %in_ptr, align 8

;   ; Create metadata for rounding mode: "round.tonearest"
;   ; error: <unknown>:0:0: ripple cannot create a vector type from this instruction's type; Allowed vector element types are integer, floating point and pointer
;   %result = call float @llvm.fptrunc.round(double %val, metadata !"round.tonearest")

;   %out_ptr = getelementptr inbounds float, ptr %dest, i64 %idx
;   store float %result, ptr %out_ptr, align 4
;   ret void
; }

define dso_local void @check_nan(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr = getelementptr inbounds float, ptr %in_array, i64 %idx
  %val = load float, ptr %in_ptr, align 4
  ; CHECK: %[[RESULT:.*]] = fcmp uno <32 x float> %val, zeroinitializer
  %is_nan = call i1 @llvm.is.fpclass.f32(float %val, i32 3)
  %result = zext i1 %is_nan to i32
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @copy_memory(ptr noundef writeonly %dest, ptr noundef readonly %src) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: [[DEST_PTR:%.*]] = extractelement <32 x ptr> %dest_ptr.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: [[SRC_PTR:%.*]] = extractelement <32 x ptr> %src_ptr.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: [[VAL:%.*]] = load i32, ptr [[SRC_PTR]]
  ; CHECK: store i32 [[VAL]], ptr [[DEST_PTR]]
  %src_ptr = getelementptr inbounds i32, ptr %src, i64 %idx
  %dest_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  call void @llvm.memcpy.p0.p0.i64(ptr %dest_ptr, ptr %src_ptr, i64 4, i1 false)
  ret void
}

define dso_local void @copy_inline(ptr noundef writeonly %dest, ptr noundef readonly %src) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: [[DEST_PTR:%.*]] = extractelement <32 x ptr> %dest_ptr.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: [[SRC_PTR:%.*]] = extractelement <32 x ptr> %src_ptr.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: [[VAL:%.*]] = load i32, ptr [[SRC_PTR]]
  ; CHECK: store i32 [[VAL]], ptr [[DEST_PTR]]
  %src_ptr = getelementptr inbounds i32, ptr %src, i64 %idx
  %dest_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  call void @llvm.memcpy.inline.p0.p0.i64(ptr %dest_ptr, ptr %src_ptr, i64 4, i1 false)
  ret void
}

define dso_local void @move_memory(ptr noundef writeonly %dest, ptr noundef readonly %src) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: [[DEST_PTR:%.*]] = extractelement <32 x ptr> %dest_ptr.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: [[SRC_PTR:%.*]] = extractelement <32 x ptr> %src_ptr.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: [[VAL:%.*]] = load i32, ptr [[SRC_PTR]]
  ; CHECK: store i32 [[VAL]], ptr [[DEST_PTR]]
  %src_ptr = getelementptr inbounds i32, ptr %src, i64 %idx
  %dest_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  call void @llvm.memmove.p0.p0.i64(ptr %dest_ptr, ptr %src_ptr, i64 4, i1 false)
  ret void
}

define dso_local void @initialize_memory(ptr noundef writeonly %dest) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: [[DEST_PTR:%.*]] = extractelement <32 x ptr> %dest_ptr.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: store i32 0, ptr [[DEST_PTR]]
  %dest_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  %dest_i8 = bitcast ptr %dest_ptr to ptr
  call void @llvm.memset.p0.i64(ptr %dest_i8, i8 0, i64 4, i1 false)
  ret void
}

define dso_local void @initialize_inline(ptr noundef writeonly %dest) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: [[DEST_PTR:%.*]] = extractelement <32 x ptr> %dest_ptr.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: store i32 0, ptr [[DEST_PTR]]
  %dest_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  %dest_i8 = bitcast ptr %dest_ptr to ptr
  call void @llvm.memset.inline.p0.i64(ptr %dest_i8, i8 0, i64 4, i1 false)
  ret void
}

define dso_local void @test_objectsize(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: %[[IN_PTR:.*]] = extractelement  <32 x ptr> %in_ptr.ripple.LS.instance, i64 %ripple.scalarcall.iterator
  ; CHECK: call i64 @llvm.objectsize.i64.p0(ptr %[[IN_PTR]], i1 false, i1 true, i1 false)
  %in_ptr = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %size = call i64 @llvm.objectsize.i64.p0(ptr %in_ptr, i1 false, i1 true, i1 false)
  %size32 = trunc i64 %size to i32
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %size32, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_sdiv_fix(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr1 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %in_ptr2 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val1 = load i32, ptr %in_ptr1, align 4
  %val2 = load i32, ptr %in_ptr2, align 4
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: %[[VAL_1:.*]] = extractelement <32 x i32> %val1, i64 %ripple.scalarcall.iterator
  ; CHECK: %[[VAL_2:.*]] = extractelement <32 x i32> %val1, i64 %ripple.scalarcall.iterator
  ; CHECK: call i32 @llvm.sdiv.fix.i32(i32 %[[VAL_1]], i32 %[[VAL_2]], i32 8)
  %result = call i32 @llvm.sdiv.fix.i32(i32 %val1, i32 %val2, i32 8)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_sdiv_fix_sat(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr1 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %in_ptr2 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val1 = load i32, ptr %in_ptr1, align 4
  %val2 = load i32, ptr %in_ptr2, align 4
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: %[[VAL_1:.*]] = extractelement <32 x i32> %val1, i64 %ripple.scalarcall.iterator
  ; CHECK: %[[VAL_2:.*]] = extractelement <32 x i32> %val1, i64 %ripple.scalarcall.iterator
  ; CHECK: call i32 @llvm.sdiv.fix.sat.i32(i32 %[[VAL_1]], i32 %[[VAL_2]], i32 8)
  %result = call i32 @llvm.sdiv.fix.sat.i32(i32 %val1, i32 %val2, i32 8)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_smul_fix(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr1 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %in_ptr2 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val1 = load i32, ptr %in_ptr1, align 4
  %val2 = load i32, ptr %in_ptr2, align 4
  ; CHECK: %[[RESULT:.*]] = call <32 x i32> @llvm.smul.fix.v32i32(<32 x i32> %val1, <32 x i32> %val1, i32 8)
  %result = call i32 @llvm.smul.fix.i32(i32 %val1, i32 %val2, i32 8)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_smul_fix_sat(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr1 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %in_ptr2 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val1 = load i32, ptr %in_ptr1, align 4
  %val2 = load i32, ptr %in_ptr2, align 4
  ; CHECK: %[[RESULT:.*]] = call <32 x i32> @llvm.smul.fix.sat.v32i32(<32 x i32> %val1, <32 x i32> %val1, i32 8)
  %result = call i32 @llvm.smul.fix.sat.i32(i32 %val1, i32 %val2, i32 8)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_udiv_fix(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr1 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %in_ptr2 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val1 = load i32, ptr %in_ptr1, align 4
  %val2 = load i32, ptr %in_ptr2, align 4
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: %[[VAL_1:.*]] = extractelement <32 x i32> %val1, i64 %ripple.scalarcall.iterator
  ; CHECK: %[[VAL_2:.*]] = extractelement <32 x i32> %val1, i64 %ripple.scalarcall.iterator
  ; CHECK: call i32 @llvm.udiv.fix.i32(i32 %[[VAL_1]], i32 %[[VAL_2]], i32 8)
  %result = call i32 @llvm.udiv.fix.i32(i32 %val1, i32 %val2, i32 8)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_udiv_fix_sat(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr1 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %in_ptr2 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val1 = load i32, ptr %in_ptr1, align 4
  %val2 = load i32, ptr %in_ptr2, align 4
  ; CHECK: .ripple.call.loop.body.call.block
  ; CHECK: %[[VAL_1:.*]] = extractelement <32 x i32> %val1, i64 %ripple.scalarcall.iterator
  ; CHECK: %[[VAL_2:.*]] = extractelement <32 x i32> %val1, i64 %ripple.scalarcall.iterator
  ; CHECK: call i32 @llvm.udiv.fix.sat.i32(i32 %[[VAL_1]], i32 %[[VAL_2]], i32 8)
  %result = call i32 @llvm.udiv.fix.sat.i32(i32 %val1, i32 %val2, i32 8)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_umul_fix(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr1 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %in_ptr2 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val1 = load i32, ptr %in_ptr1, align 4
  %val2 = load i32, ptr %in_ptr2, align 4
  ; CHECK: %[[RESULT:.*]] = call <32 x i32> @llvm.umul.fix.v32i32(<32 x i32> %val1, <32 x i32> %val1, i32 8)
  %result = call i32 @llvm.umul.fix.i32(i32 %val1, i32 %val2, i32 8)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

define dso_local void @test_umul_fix_sat(ptr noundef writeonly %dest, ptr noundef readonly %in_array) {
entry:
  %BS = tail call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 32, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %idx = tail call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  %in_ptr1 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %in_ptr2 = getelementptr inbounds i32, ptr %in_array, i64 %idx
  %val1 = load i32, ptr %in_ptr1, align 4
  %val2 = load i32, ptr %in_ptr2, align 4
  ; CHECK: %[[RESULT:.*]] = call <32 x i32> @llvm.umul.fix.sat.v32i32(<32 x i32> %val1, <32 x i32> %val1, i32 8)
  %result = call i32 @llvm.umul.fix.sat.i32(i32 %val1, i32 %val2, i32 8)
  %out_ptr = getelementptr inbounds i32, ptr %dest, i64 %idx
  store i32 %result, ptr %out_ptr, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.abs.i32(i32, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.convert.from.fp16.f32(i16) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.cttz.i32(i32, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f32(float, i32 immarg) #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.inline.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.inline.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.objectsize.i64.p0(ptr, i1 immarg, i1 immarg, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.sdiv.fix.i32(i32, i32, i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.sdiv.fix.sat.i32(i32, i32, i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smul.fix.i32(i32, i32, i32 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smul.fix.sat.i32(i32, i32, i32 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.udiv.fix.i32(i32, i32, i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.udiv.fix.sat.i32(i32, i32, i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umul.fix.i32(i32, i32, i32 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umul.fix.sat.i32(i32, i32, i32 immarg) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
