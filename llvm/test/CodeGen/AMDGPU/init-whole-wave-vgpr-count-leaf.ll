; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 < %s | FileCheck %s

; CHECK-LABEL: .shader_functions:

; Make sure that .vgpr_count doesn't include the %inactive.vgpr registers.
; CHECK-LABEL: leaf_shader:
; CHECK: .vgpr_count: 0xc{{$}}

; Function without calls.
define amdgpu_cs_chain void @_leaf_shader(ptr %output.ptr, i32 inreg %input.value,
                              i32 %active.vgpr1, i32 %active.vgpr2,
                              i32 %inactive.vgpr1, i32 %inactive.vgpr2, i32 %inactive.vgpr3,
                              i32 %inactive.vgpr4, i32 %inactive.vgpr5, i32 %inactive.vgpr6)
                              local_unnamed_addr {
entry:
  %dead.val = call i32 @llvm.amdgcn.dead.i32()
  %is.whole.wave = call i1 @llvm.amdgcn.init.whole.wave()
  br i1 %is.whole.wave, label %compute, label %merge

compute:
  ; Perform a more complex computation using active VGPRs
  %square = mul i32 %active.vgpr1, %active.vgpr1
  %product = mul i32 %square, %active.vgpr2
  %sum = add i32 %product, %input.value
  %result = add i32 %sum, 42
  br label %merge

merge:
  %final.result = phi i32 [ 0, %entry ], [ %result, %compute ]
  %final.inactive1 = phi i32 [ %inactive.vgpr1, %entry ], [ %dead.val, %compute ]
  %final.inactive2 = phi i32 [ %inactive.vgpr2, %entry ], [ %dead.val, %compute ]
  %final.inactive3 = phi i32 [ %inactive.vgpr3, %entry ], [ %dead.val, %compute ]
  %final.inactive4 = phi i32 [ %inactive.vgpr4, %entry ], [ %dead.val, %compute ]
  %final.inactive5 = phi i32 [ %inactive.vgpr5, %entry ], [ %dead.val, %compute ]
  %final.inactive6 = phi i32 [ %inactive.vgpr6, %entry ], [ %dead.val, %compute ]

  store i32 %final.result, ptr %output.ptr, align 4

  ret void
}

declare i32 @llvm.amdgcn.dead.i32()
declare i1 @llvm.amdgcn.init.whole.wave()
declare void @llvm.amdgcn.cs.chain.p0.i32.v4i32.sl_i32i32i32i32i32i32i32i32i32i32i32i32s(ptr, i32, <4 x i32>, { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, i32 immarg, ...)

declare amdgpu_cs_chain void @retry_vgpr_alloc.v4i32(<4 x i32> inreg)
