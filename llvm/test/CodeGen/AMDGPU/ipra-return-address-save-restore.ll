; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -enable-ipra=1 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -enable-ipra=0 < %s | FileCheck -check-prefix=GCN %s

; This test is to make sure the return address registers, if clobbered in the
; function or the function has calls, are save/restored when IPRA is enabled/disabled.

; TODO: An artificial test with high register pressure would be more reliable in the
; long run as branches on constants could be fragile.

%struct.ShaderData = type { <3 x float>, <3 x float>, <3 x float>, <3 x float>, i32, i32, i32, i32, i32, float, float, i32, i32, float, float, %struct.differential3, %struct.differential3, %struct.differential, %struct.differential, <3 x float>, <3 x float>, <3 x float>, %struct.differential3, i32, i32, i32, float, <3 x float>, <3 x float>, <3 x float>, [1 x %struct.ShaderClosure] }
%struct.differential = type { float, float }
%struct.differential3 = type { <3 x float>, <3 x float> }
%struct.ShaderClosure = type { <3 x float>, i32, float, <3 x float>, [10 x float], [8 x i8] }
%struct.MicrofacetExtra = type { <3 x float>, <3 x float>, <3 x float>, float, [12 x i8] }

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <3 x float> @llvm.fmuladd.v3f32(<3 x float>, <3 x float>, <3 x float>) #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #0

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p5i8(i64 immarg, i8 addrspace(5)* nocapture) #1

; Function Attrs: norecurse
define internal fastcc void @svm_node_closure_bsdf(%struct.ShaderData addrspace(1)* %sd, float* %stack, <4 x i32> %node, i32* %offset, i32 %0, i8 %trunc, float %1, float %2, float %mul80, i1 %cmp412.old, <4 x i32> %3, float %4, i32 %5, i1 %cmp440, i1 %cmp442, i1 %or.cond1306, float %.op, %struct.ShaderClosure addrspace(1)* %arrayidx.i.i2202, %struct.ShaderClosure addrspace(1)* %retval.0.i.i22089, %struct.ShaderClosure addrspace(1)* %retval.1.i221310, i1 %cmp575, i32 addrspace(1)* %num_closure_left.i2215, i32 %6, i1 %cmp.i2216, i32 %7, i64 %idx.ext.i2223, i32 %sub5.i2221) #2 {
; GCN-LABEL: {{^}}svm_node_closure_bsdf:
; GCN-DAG: v_writelane_b32 [[CSR_VGPR:v[0-9]+]], s30,
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s31,
; GCN: s_movk_i32 s30, 0x60
; GCN-NOT: s31
; GCN-DAG: v_readlane_b32 s31, [[CSR_VGPR]],
; GCN-DAG: v_readlane_b32 s30, [[CSR_VGPR]],
; GCN: s_waitcnt vmcnt(0)
; GCN: s_setpc_b64 s[30:31]
entry:
  %8 = extractelement <4 x i32> %node, i64 0
  %cmp.i.not = icmp eq i32 undef, 0
  br i1 undef, label %common.ret.critedge, label %cond.true

cond.true:                                        ; preds = %entry
  %9 = load float, float* null, align 4
  %phi.cmp = fcmp oeq float %9, 0.000000e+00
  br i1 %phi.cmp, label %common.ret, label %cond.true20

cond.true20:                                      ; preds = %cond.true
  %trunc1 = trunc i32 %0 to i8
  switch i8 %trunc, label %common.ret [
    i8 44, label %sw.bb
    i8 0, label %if.end.i.i2285
  ]

sw.bb:                                            ; preds = %cond.true20
  %10 = load float, float* null, align 4
  %11 = load float, float* null, align 4
  %12 = tail call float @llvm.amdgcn.fmed3.f32(float %1, float 0.000000e+00, float 0.000000e+00)
  %mul802 = fmul nsz float %1, 0.000000e+00
  %cmp412.old3 = fcmp nsz ogt float %1, 0.000000e+00
  br i1 %cmp412.old, label %if.then413, label %common.ret

if.then413:                                       ; preds = %sw.bb
  %13 = load <4 x i32>, <4 x i32> addrspace(1)* null, align 16
  %14 = extractelement <4 x i32> %node, i64 0
  %cmp4404 = fcmp nsz ole float %1, 0.000000e+00
  %cmp4425 = icmp eq i32 %0, 0
  %or.cond13066 = select i1 %cmp412.old, i1 false, i1 %cmp412.old
  br i1 %or.cond1306, label %if.then443, label %if.else568

if.then443:                                       ; preds = %if.then413
  br i1 true, label %if.end511, label %common.ret

common.ret.critedge:                              ; preds = %entry
  store i32 0, i32* null, align 4
  br label %common.ret

common.ret:                                       ; preds = %if.end.i.i2285, %if.end627.sink.split, %cond.end579, %bsdf_alloc.exit2188, %if.end511, %common.ret.critedge, %if.then443, %sw.bb, %cond.true20, %cond.true
  ret void

if.end511:                                        ; preds = %if.then443
  br i1 false, label %common.ret, label %if.then519

if.then519:                                       ; preds = %if.end511
  br i1 false, label %bsdf_alloc.exit2188, label %if.then.i2172

if.then.i2172:                                    ; preds = %if.then519
  br i1 false, label %closure_alloc.exit.i2184, label %if.end.i.i2181

if.end.i.i2181:                                   ; preds = %if.then.i2172
  br label %closure_alloc.exit.i2184

closure_alloc.exit.i2184:                         ; preds = %if.end.i.i2181, %if.then.i2172
  br i1 false, label %bsdf_alloc.exit2188, label %if.end.i2186

if.end.i2186:                                     ; preds = %closure_alloc.exit.i2184
  br label %bsdf_alloc.exit2188

bsdf_alloc.exit2188:                              ; preds = %if.end.i2186, %closure_alloc.exit.i2184, %if.then519
  br i1 false, label %common.ret, label %if.then534

if.then534:                                       ; preds = %bsdf_alloc.exit2188
  %.op7 = fmul nsz float undef, 0.000000e+00
  %mul558 = select i1 %cmp440, float 0.000000e+00, float %1
  %15 = tail call float @llvm.amdgcn.fmed3.f32(float 0.000000e+00, float 0.000000e+00, float 0.000000e+00)
  store float %mul558, float addrspace(1)* null, align 4
  br label %if.end627.sink.split

if.else568:                                       ; preds = %if.then413
  br i1 undef, label %bsdf_alloc.exit2214, label %if.then.i2198

if.then.i2198:                                    ; preds = %if.else568
  br i1 undef, label %closure_alloc.exit.i2210, label %if.end.i.i2207

if.end.i.i2207:                                   ; preds = %if.then.i2198
  %arrayidx.i.i22028 = getelementptr inbounds %struct.ShaderData, %struct.ShaderData addrspace(1)* %sd, i64 0, i32 30, i64 undef
  br label %closure_alloc.exit.i2210

closure_alloc.exit.i2210:                         ; preds = %if.end.i.i2207, %if.then.i2198
  %retval.0.i.i220899 = phi %struct.ShaderClosure addrspace(1)* [ %arrayidx.i.i2202, %if.end.i.i2207 ], [ null, %if.then.i2198 ]
  br i1 false, label %bsdf_alloc.exit2214, label %if.end.i2212

if.end.i2212:                                     ; preds = %closure_alloc.exit.i2210
  br label %bsdf_alloc.exit2214

bsdf_alloc.exit2214:                              ; preds = %if.end.i2212, %closure_alloc.exit.i2210, %if.else568
  %retval.1.i22131010 = phi %struct.ShaderClosure addrspace(1)* [ %arrayidx.i.i2202, %if.end.i2212 ], [ null, %closure_alloc.exit.i2210 ], [ null, %if.else568 ]
  %cmp57511 = icmp ne %struct.ShaderClosure addrspace(1)* %arrayidx.i.i2202, null
  br i1 %cmp442, label %cond.true576, label %cond.end579

cond.true576:                                     ; preds = %bsdf_alloc.exit2214
  %num_closure_left.i221512 = getelementptr inbounds %struct.ShaderData, %struct.ShaderData addrspace(1)* %sd, i64 0, i32 25
  %16 = load i32, i32 addrspace(1)* %num_closure_left.i2215, align 8
  %cmp.i221613 = icmp slt i32 %0, 0
  br i1 %cmp440, label %cond.end579, label %if.end.i2227

if.end.i2227:                                     ; preds = %cond.true576
  %sub5.i222114 = add nuw nsw i32 %0, 0
  %17 = load i32, i32 addrspace(1)* null, align 4294967296
  %idx.ext.i222315 = sext i32 %0 to i64
  %add.ptr.i2224 = getelementptr inbounds %struct.ShaderData, %struct.ShaderData addrspace(1)* %sd, i64 0, i32 30, i64 %idx.ext.i2223
  %idx.ext8.i22252724 = zext i32 %0 to i64
  %add.ptr9.i2226 = getelementptr inbounds %struct.ShaderClosure, %struct.ShaderClosure addrspace(1)* %add.ptr.i2224, i64 %idx.ext8.i22252724
  %phi.cast2731 = bitcast %struct.ShaderClosure addrspace(1)* %add.ptr9.i2226 to %struct.MicrofacetExtra addrspace(1)*
  br label %cond.end579

cond.end579:                                      ; preds = %if.end.i2227, %cond.true576, %bsdf_alloc.exit2214
  %cond580 = phi %struct.MicrofacetExtra addrspace(1)* [ null, %bsdf_alloc.exit2214 ], [ %phi.cast2731, %if.end.i2227 ], [ null, %cond.true576 ]
  %tobool583 = icmp ne %struct.MicrofacetExtra addrspace(1)* %cond580, null
  %or.cond1308 = select i1 %cmp442, i1 %tobool583, i1 false
  br i1 %or.cond1308, label %if.then584, label %common.ret

if.then584:                                       ; preds = %cond.end579
  store %struct.MicrofacetExtra addrspace(1)* null, %struct.MicrofacetExtra addrspace(1)* addrspace(1)* null, align 4294967296
  br label %if.end627.sink.split

if.end627.sink.split:                             ; preds = %if.then584, %if.then534
  store i32 0, i32 addrspace(1)* null, align 4
  br label %common.ret

if.end.i.i2285:                                   ; preds = %cond.true20
  store i32 0, i32 addrspace(1)* null, align 4294967296
  br label %common.ret
}

define internal fastcc void @svm_eval_nodes(%struct.ShaderData addrspace(1)* %sd) {
sw.bb10:
; GCN-LABEL: {{^}}svm_eval_nodes:
; GCN-DAG: v_writelane_b32 [[CSR_VGPR:v[0-9]+]], s30,
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s31,
; GCN: s_swappc_b64 s[30:31]
; GCN-DAG: v_readlane_b32 s31, [[CSR_VGPR]],
; GCN-DAG: v_readlane_b32 s30, [[CSR_VGPR]],
; GCN: s_waitcnt vmcnt(0)
; GCN: s_setpc_b64 s[30:31]
  call fastcc void @svm_node_closure_bsdf(%struct.ShaderData addrspace(1)* null, float* null, <4 x i32> zeroinitializer, i32* null, i32 undef, i8 undef, float undef, float undef, float undef, i1 undef, <4 x i32> undef, float undef, i32 undef, i1 undef, i1 undef, i1 undef, float undef, %struct.ShaderClosure addrspace(1)* undef, %struct.ShaderClosure addrspace(1)* undef, %struct.ShaderClosure addrspace(1)* undef, i1 undef, i32 addrspace(1)* undef, i32 undef, i1 undef, i32 undef, i64 undef, i32 undef)
  ret void
}

define amdgpu_kernel void @kernel_ocl_path_trace_shadow_blocked_dl() {
kernel_set_buffer_pointers.exit:
; GCN-LABEL: {{^}}kernel_ocl_path_trace_shadow_blocked_dl:
; GCN: s_swappc_b64 s[30:31]
; GCN: endpgm
  tail call fastcc void @svm_eval_nodes(%struct.ShaderData addrspace(1)* null)
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fabs.f32(float) #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.maxnum.f32(float, float) #0

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.amdgcn.fmed3.f32(float, float, float) #3

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { norecurse }
attributes #3 = { nounwind readnone speculatable willreturn }
