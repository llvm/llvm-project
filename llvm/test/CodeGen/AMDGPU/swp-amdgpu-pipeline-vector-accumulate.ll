; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-enable-pipeliner -pass-remarks-analysis=pipeliner %s -filetype=null 2>&1 | FileCheck %s
; Reduced from Composable Kernels. Vector <2 x float> reduction with
; two accumulators fed by two buffer loads and vector-predicated selects.
; Verifies such a loop software-pipelines.
; CHECK: Schedule found with Initiation Interval

define amdgpu_kernel void @swp_amdgpu_pipeline_vector_accumulate(i32 %arg, ptr addrspace(8) %arg1, i1 %arg2, i1 %arg3) {
bb:
  %call = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb4

bb4:                                              ; preds = %bb4, %bb
  %phi = phi i32 [ %call, %bb ], [ %or, %bb4 ]
  %phi5 = phi i32 [ %call, %bb ], [ %add, %bb4 ]
  %phi6 = phi <2 x float> [ zeroinitializer, %bb ], [ %fadd, %bb4 ]
  %phi7 = phi <2 x float> [ zeroinitializer, %bb ], [ %fadd19, %bb4 ]
  %phi8 = phi i32 [ 0, %bb ], [ 2, %bb4 ]
  %shl = shl i32 %phi5, 1
  %call9 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %arg1, i32 %shl, i32 0, i32 0)
  %insertelement = insertelement <2 x i32> zeroinitializer, i32 %call9, i64 0
  %icmp = icmp slt i32 %phi, 0
  %call10 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) null, i32 %shl, i32 0, i32 0)
  %and = and i1 %arg2, %icmp
  %insertelement11 = insertelement <2 x i1> zeroinitializer, i1 %and, i64 0
  %and12 = and i1 %arg3, %icmp
  %insertelement13 = insertelement <2 x i1> %insertelement11, i1 %and12, i64 1
  %insertelement14 = insertelement <2 x i32> %insertelement, i32 1, i64 1
  %bitcast = bitcast <2 x i32> %insertelement14 to <2 x float>
  %select = select <2 x i1> %insertelement13, <2 x float> %bitcast, <2 x float> zeroinitializer
  %fadd = fadd <2 x float> %phi6, %select
  %insertelement15 = insertelement <2 x i32> zeroinitializer, i32 %call10, i64 0
  %insertelement16 = insertelement <2 x i32> %insertelement15, i32 %arg, i64 1
  %bitcast17 = bitcast <2 x i32> %insertelement16 to <2 x float>
  %select18 = select <2 x i1> %insertelement13, <2 x float> %bitcast17, <2 x float> zeroinitializer
  %fadd19 = fadd <2 x float> %phi7, %select18
  %or = or i32 %phi, 1
  %add = add i32 %phi5, 1
  %icmp20 = icmp eq i32 %phi8, 0
  br i1 %icmp20, label %bb21, label %bb4

bb21:                                             ; preds = %bb4
  ret void
}
