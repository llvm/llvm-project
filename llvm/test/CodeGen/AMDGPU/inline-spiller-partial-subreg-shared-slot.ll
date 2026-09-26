; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stop-after=greedy %s -o - | FileCheck %s
;
; Regression test for the InlineSpiller shared-stack-slot partial-subregister
; hazard.
;
; The hazard is target-independent: InlineSpiller's shared-slot logic is generic,
; and SGPR-spill-to-VGPR-lane exists on every AMDGPU that spills SGPRs. When the
; greedy allocator live-range-splits an sgpr_128 descriptor on one sub-register,
; it produces sibling vregs each defined on a single sub-lane
; (`undef %V.sub0:sgpr_128 = ...`), undef elsewhere. Since InlineSpiller shares
; one stack slot among descendants of the Original value, a full-width store of
; such a sibling writes its undef lanes over a live sibling in the SAME slot. This
; input is a gfx950 reproducer (there the clobber corrupts an in-loop buffer
; descriptor -- the kind of corruption that leads to an HSA memory fault), but the
; fix and this check apply to all AMDGPU.
;
; The fix records in the save pseudo's $lanemask operand which dwords the store
; defines; spillSGPR writelanes only those, leaving the slot's other lanes intact.
; This pins that the two partial siblings sharing one slot each store with a
; partial mask (matched as a non-negative immediate: the buggy full-width store
; used -1, which the {{[0-9]+}} pattern cannot match).
;
; Driven from IR through -stop-after=greedy: a .mir + -run-pass=greedy form does
; not reproduce, since the trigger needs pre-greedy PreRARemat pressure state that
; MIR serialization drops. The CHECK block is vreg/slot-number independent.
;
; First partial sibling (only .sub0 defined), stored with a partial mask:
; CHECK:      undef [[DESC:%[0-9]+]].sub0:sgpr_128 = S_MOV_B32 0
; CHECK-NEXT: SI_SPILL_S128_SAVE [[DESC]], [[SLOT:%stack\.[0-9]+]], {{[0-9]+}},
; Second partial sibling into the SAME slot, also with a partial mask:
; CHECK:      SI_SPILL_S128_SAVE %{{[0-9]+}}, [[SLOT]], {{[0-9]+}},
;
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @_attn_fwd_IS_CAUSAL_1_NUM_Q_HEADS_8_NUM_K_HEADS_1_BLOCK_M_256_BLOCK_N_64_BLOCK_DMODEL_64_RETURN_SCORES_1_ENABLE_DROPOUT_1_IS_FP8_0_VARLEN_1_NUM_XCD_8_USE_INT64_STRIDES_1_ENABLE_SINK_0_SLIDING_WINDOW_0(ptr addrspace(1) inreg %0, ptr addrspace(1) inreg %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, <1 x i32> %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14, i32 %15, i32 %16, i32 %17, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, i64 %37, i1 %38, i1 %39, i1 %40, i1 %41, i1 %42, i1 %43, i1 %44, i64 %sext452, i64 %.pn241.in795, i64 %sext496, i64 %.pn233.in799, i64 %sext500, i64 %.pn231.in800, i64 %sext501, i64 %.pn229.in801, i64 %sext503, i64 %.pn223.in804, i64 %sext505, i64 %.pn221.in805, i64 %sext506) #0 {
..loopexit_crit_edge:
  %45 = icmp slt i32 %8, 0
  %46 = icmp slt i32 %6, 0
  %47 = icmp slt i32 %9, 0
  %48 = icmp slt i32 %10, 0
  %49 = icmp slt i32 %12, 0
  %50 = icmp slt i32 %2, 0
  br label %51

51:                                               ; preds = %51, %..loopexit_crit_edge
  %.pn265.in835 = phi i64 [ 0, %..loopexit_crit_edge ], [ %sext503, %51 ]
  %.pn179.in826 = phi i64 [ 0, %..loopexit_crit_edge ], [ %27, %51 ]
  %.pn181.in8251 = phi i64 [ 0, %..loopexit_crit_edge ], [ %137, %51 ]
  %.pn183.in824 = phi i64 [ 0, %..loopexit_crit_edge ], [ %sext506, %51 ]
  %.pn185.in8232 = phi i64 [ 0, %..loopexit_crit_edge ], [ %sext501, %51 ]
  %.pn195.in818 = phi i64 [ 0, %..loopexit_crit_edge ], [ %135, %51 ]
  %.pn197.in8177 = phi i64 [ 0, %..loopexit_crit_edge ], [ %sext452, %51 ]
  %.pn199.in8168 = phi i64 [ 0, %..loopexit_crit_edge ], [ %.pn229.in801, %51 ]
  %.pn201.in8159 = phi i64 [ 0, %..loopexit_crit_edge ], [ %30, %51 ]
  %.pn203.in81410 = phi i64 [ 0, %..loopexit_crit_edge ], [ %31, %51 ]
  %.pn205.in813 = phi i64 [ 0, %..loopexit_crit_edge ], [ 2, %51 ]
  %.pn209.in811 = phi i64 [ 0, %..loopexit_crit_edge ], [ 1, %51 ]
  %.pn215.in80811 = phi i64 [ 0, %..loopexit_crit_edge ], [ %.pn231.in800, %51 ]
  %.pn217.in80712 = phi i64 [ 0, %..loopexit_crit_edge ], [ %32, %51 ]
  %.pn219.in80613 = phi i64 [ 0, %..loopexit_crit_edge ], [ %33, %51 ]
  %.pn223.in80415 = phi i64 [ 0, %..loopexit_crit_edge ], [ %136, %51 ]
  %.pn227.in80216 = phi i64 [ 0, %..loopexit_crit_edge ], [ %sext505, %51 ]
  %.pn229.in80117 = phi i64 [ 0, %..loopexit_crit_edge ], [ %sext496, %51 ]
  %.pn231.in80018 = phi i64 [ 0, %..loopexit_crit_edge ], [ %35, %51 ]
  %.pn233.in79919 = phi i64 [ 0, %..loopexit_crit_edge ], [ %36, %51 ]
  %.pn241.in79520 = phi i64 [ 0, %..loopexit_crit_edge ], [ %37, %51 ]
  %52 = phi i32 [ 0, %..loopexit_crit_edge ], [ 2, %51 ]
  %53 = phi <2 x float> [ zeroinitializer, %..loopexit_crit_edge ], [ %138, %51 ]
  %54 = or i32 %52, %20
  %55 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %1, i16 0, i64 1, i32 1)
  %.pn265 = trunc i64 %.pn265.in835 to i32
  %56 = shl i32 %.pn265, 1
  %57 = tail call <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) %55, i32 %56, i32 0, i32 0)
  %58 = icmp slt i32 %52, %5
  %59 = select i1 %58, i32 0, i32 1
  %60 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) null, i32 %59, i32 0, i32 0)
  %61 = icmp slt i32 %54, 0
  %62 = and i1 %46, %61
  %63 = and i1 %47, %61
  %64 = and i1 %48, %61
  %65 = icmp slt i32 %3, 0
  %66 = and i1 %65, %61
  %67 = icmp slt i32 %13, 0
  %68 = and i1 %67, %61
  %69 = icmp slt i32 %14, 0
  %70 = and i1 %69, %61
  %71 = and i1 %38, %61
  %72 = and i1 %50, %61
  %73 = and i1 %39, %61
  %74 = icmp slt i32 %23, 0
  %75 = and i1 %74, %61
  %76 = and i1 %49, %61
  %77 = icmp slt i32 %25, 0
  %78 = and i1 %77, %61
  %79 = icmp slt i32 %18, 0
  %80 = and i1 %79, %61
  %81 = and i1 %40, %61
  %82 = icmp slt i32 %24, 0
  %83 = and i1 %82, %61
  store <2 x i32> %57, ptr addrspace(3) null, align 8
  tail call void @llvm.amdgcn.s.barrier()
  %84 = icmp slt i32 %52, 1
  %85 = and i1 %45, %84
  %.pn241 = trunc i64 %.pn241.in79520 to i32
  %86 = select i1 %62, i32 %.pn241, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %86, i32 0, i32 0)
  %87 = select i1 %63, i32 %3, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %87, i32 0, i32 0)
  %88 = select i1 %64, i32 1, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %88, i32 0, i32 0)
  %89 = and i1 %44, %61
  %.pn239 = trunc i64 %.pn209.in811 to i32
  %90 = select i1 %89, i32 %.pn239, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %90, i32 0, i32 0)
  %91 = and i1 %39, %38
  %.pn233 = trunc i64 %.pn233.in79919 to i32
  %92 = select i1 %91, i32 %.pn233, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %92, i32 0, i32 0)
  %.pn231 = trunc i64 %.pn231.in80018 to i32
  %93 = select i1 %39, i32 %.pn231, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %93, i32 0, i32 0)
  %.pn229 = trunc i64 %.pn229.in80117 to i32
  %94 = select i1 %38, i32 %.pn229, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %94, i32 0, i32 0)
  %.pn227 = trunc i64 %.pn227.in80216 to i32
  %95 = select i1 %39, i32 %.pn227, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %95, i32 0, i32 0)
  %96 = select i1 %66, i32 0, i32 1
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %96, i32 0, i32 0)
  %.pn223 = trunc i64 %.pn223.in80415 to i32
  %97 = shl i32 %.pn223, 1
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %97, i32 0, i32 0)
  %98 = select i1 %68, i32 %14, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %98, i32 0, i32 0)
  %.pn219 = trunc i64 %.pn219.in80613 to i32
  %99 = select i1 %70, i32 %.pn219, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %99, i32 0, i32 0)
  %100 = and i1 %43, %38
  %.pn217 = trunc i64 %.pn217.in80712 to i32
  %101 = select i1 %100, i32 %.pn217, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %101, i32 0, i32 0)
  %.pn215 = trunc i64 %.pn215.in80811 to i32
  %102 = select i1 %39, i32 %.pn215, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %102, i32 0, i32 0)
  %.pn213 = trunc i64 %.pn205.in813 to i32
  %103 = select i1 %38, i32 %.pn213, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %103, i32 0, i32 0)
  %104 = select i1 %71, i32 1, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %104, i32 0, i32 0)
  %105 = select i1 %72, i32 1, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %105, i32 0, i32 0)
  %.pn207 = trunc i64 %28 to i32
  %106 = select i1 %73, i32 %.pn207, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %106, i32 0, i32 0)
  %107 = select i1 %75, i32 %16, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %107, i32 0, i32 0)
  %.pn203 = trunc i64 %.pn203.in81410 to i32
  %108 = select i1 %76, i32 %.pn203, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %108, i32 0, i32 0)
  %109 = and i1 %41, %38
  %.pn201 = trunc i64 %.pn201.in8159 to i32
  %110 = select i1 %109, i32 %.pn201, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %110, i32 0, i32 0)
  %.pn199 = trunc i64 %.pn199.in8168 to i32
  %111 = shl i32 %.pn199, 0
  %112 = select i1 %38, i32 %111, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %112, i32 0, i32 0)
  %.pn197 = trunc i64 %.pn197.in8177 to i32
  %113 = select i1 %78, i32 %.pn197, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %113, i32 0, i32 0)
  %114 = and i1 %38, %40
  %.pn195 = trunc i64 %.pn195.in818 to i32
  %115 = select i1 %114, i32 %.pn195, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %115, i32 0, i32 0)
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %54, i32 0, i32 0)
  %116 = select i1 %80, i32 %21, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %116, i32 0, i32 0)
  %.pn189 = trunc i64 %.pn179.in826 to i32
  %117 = select i1 %42, i32 %.pn189, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %117, i32 0, i32 0)
  %118 = select i1 %81, i32 %.pn265, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %118, i32 0, i32 0)
  %.pn185 = trunc i64 %.pn185.in8232 to i32
  %119 = select i1 %50, i32 %.pn185, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %119, i32 0, i32 0)
  %.pn183 = trunc i64 %.pn183.in824 to i32
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %.pn183, i32 0, i32 0)
  %.pn181 = trunc i64 %.pn181.in8251 to i32
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %.pn181, i32 0, i32 0)
  %120 = select i1 %83, i32 %3, i32 0
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %120, i32 0, i32 0)
  %121 = select i1 %85, i32 0, i32 -2147483648
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float 0.000000e+00, ptr addrspace(8) null, i32 %121, i32 0, i32 0)
  %122 = bitcast <4 x i32> %60 to <8 x bfloat>
  %123 = shufflevector <8 x bfloat> %122, <8 x bfloat> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x bfloat> %123, ptr addrspace(3) null, align 8
  %124 = shufflevector <2 x float> zeroinitializer, <2 x float> %53, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %125 = shufflevector <16 x float> %124, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %126 = shufflevector <16 x float> %125, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %127 = shufflevector <16 x float> %126, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %128 = shufflevector <16 x float> %127, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison>
  %129 = shufflevector <16 x float> %128, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 poison, i32 poison>
  %130 = shufflevector <16 x float> %129, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  %131 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> splat (bfloat 1.000000e+00), <16 x float> %130, i32 0, i32 0, i32 0)
  %132 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> splat (bfloat 1.000000e+00), <16 x float> %131, i32 0, i32 0, i32 0)
  %133 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> splat (bfloat +qnan), <8 x bfloat> zeroinitializer, <16 x float> %132, i32 0, i32 0, i32 0)
  %134 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> splat (bfloat +qnan), <8 x bfloat> splat (bfloat 1.000000e+00), <16 x float> %133, i32 0, i32 0, i32 0)
  %135 = ashr i64 %27, 1
  %136 = or i64 %.pn233.in799, 1
  %137 = or i64 %sext500, 1
  %138 = shufflevector <16 x float> %134, <16 x float> zeroinitializer, <2 x i32> <i32 2, i32 3>
  br label %51
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, i32 immarg) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: read)
declare <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) captures(none)) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) readnone, i16, i64, i32) #6

; uselistorder directives
uselistorder ptr @llvm.amdgcn.raw.ptr.buffer.store.f32, { 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
uselistorder ptr @llvm.amdgcn.mfma.f32.32x32x16.bf16, { 3, 2, 1, 0 }

attributes #0 = { "amdgpu-agpr-alloc"="0" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-queue-ptr" "amdgpu-waves-per-eu"="8,8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { convergent nocallback nofree nounwind willreturn }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(argmem: read) }
