; RUN: opt -S -passes=dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

@A.str = internal unnamed_addr constant [2 x i8] c"A\00", align 1
@B.str = internal unnamed_addr constant [2 x i8] c"A\00", align 1

declare i32 @some_val();

define void @test_buffers_with_nuri() {

  %val = call i32 @some_val()
  %foo = alloca i32, align 4

  ; RWBuffer<float> A[10];
  ;
  ; A[NonUniformResourceIndex(val)];

  %nuri1 = tail call noundef i32 @llvm.dx.resource.nonuniformindex(i32 %val)
  %res1 = call target("dx.TypedBuffer", float, 1, 0, 0) 
            @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 10, i32 %nuri1, ptr @A.str)
  ; CHECK: %[[RES1:.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 0, i32 9, i32 0, i8 1 }, i32 %val, i1 true) #[[ATTR:.*]]
  ; CHECK: call  %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %[[RES1]], %dx.types.ResourceProperties { i32 4106, i32 265 }) #[[ATTR]]
  ; CHECK-NOT: @llvm.dx.cast.handle
  ; CHECK-NOT: @llvm.dx.resource.nonuniformindex

  ; A[NonUniformResourceIndex(val + 1) % 10];
  %add1 = add i32 %val, 1
  %nuri2 = tail call noundef i32 @llvm.dx.resource.nonuniformindex(i32 %add1)
  %rem1 = urem i32 %nuri2, 10
  %res2 = call target("dx.TypedBuffer", float, 1, 0, 0) 
           @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 10, i32 %rem1, ptr @A.str)
  ; CHECK: %[[RES2:.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 0, i32 9, i32 0, i8 1 }, i32 %rem1, i1 true) #[[ATTR]]
  ; CHECK: call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %[[RES2]], %dx.types.ResourceProperties { i32 4106, i32 265 }) #[[ATTR]]

  ; A[10 + 3 * NonUniformResourceIndex(GI)];
  %mul1 = mul i32 %nuri1, 3
  %add2 = add i32 %mul1, 10
  %res3 = call target("dx.TypedBuffer", float, 1, 0, 0)
           @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 10, i32 %add2, ptr @A.str)
  ; CHECK: %[[RES3:.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 0, i32 9, i32 0, i8 1 }, i32 %add2, i1 true) #[[ATTR]]
  ; CHECK: %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %[[RES3]], %dx.types.ResourceProperties { i32 4106, i32 265 }) #[[ATTR]]
  ret void

  ; NonUniformResourceIndex value going through store & load: the flag is not going to get picked up
  %a = tail call noundef i32 @llvm.dx.resource.nonuniformindex(i32 %val)
  store i32 %a, ptr %foo
  %b = load i32, ptr %foo
  %res4 = call target("dx.TypedBuffer", float, 1, 0, 0)
           @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 10, i32 %b, ptr @A.str)
  ; CHECK: %[[RES4:.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 0, i32 9, i32 0, i8 1 }, i32 %b, i1 false) #[[ATTR]]
  ; CHECK: %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %[[RES4]], %dx.types.ResourceProperties { i32 4106, i32 265 }) #[[ATTR]]

  ; NonUniformResourceIndex index value on a single resouce (not an array): the flag is not going to get picked up
  ; RWBuffer<float> B : register(u20);
  ;
  ; B[NonUniformResourceIndex(val)];

  %nuri3 = tail call noundef i32 @llvm.dx.resource.nonuniformindex(i32 %val)
  %res5 = call target("dx.TypedBuffer", float, 1, 0, 0)
           @llvm.dx.resource.handlefrombinding(i32 20, i32 0, i32 1, i32 %nuri1, ptr @B.str)
  ; CHECK: %[[RES4:.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 0, i32 0, i32 20, i8 1 }, i32 %val, i1 false) #[[ATTR]]
  ; CHECK: %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %[[RES4]], %dx.types.ResourceProperties { i32 4106, i32 265 }) #[[ATTR]]

  ; NonUniformResourceIndex on unrelated value - the call is removed:
  ; foo = NonUniformResourceIndex(val);
  %nuri4 = tail call noundef i32 @llvm.dx.resource.nonuniformindex(i32 %val)
  store i32 %nuri4, ptr %foo
  ; CHECK: store i32 %val, ptr %foo
  ; CHECK-NOT: @llvm.dx.resource.nonuniformindex

  ret void
}

; CHECK: attributes #[[ATTR]] = {{{.*}} memory(none) {{.*}}}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
