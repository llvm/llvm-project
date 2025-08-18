; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 < %s | FileCheck %s

; CHECK-LABEL: .shader_functions:

; Make sure that .vgpr_count doesn't include the %inactive.vgpr registers.
; The shader is free to use any of the VGPRs mapped to a %inactive.vgpr as long as it only touches its active lanes.
; In that case, the VGPR should be included in the .vgpr_count
; CHECK-LABEL: _miss_1:
; CHECK: .vgpr_count: 0xd{{$}}

define amdgpu_cs_chain void @_miss_1(ptr inreg %next.callee, i32 inreg %global.table, i32 inreg %max.outgoing.vgpr.count,
                                    i32 %vcr, { i32 } %system.data,
                                    i32 %inactive.vgpr, i32 %inactive.vgpr1, i32 %inactive.vgpr2, i32 %inactive.vgpr3,
                                    i32 %inactive.vgpr4, i32 %inactive.vgpr5, i32 %inactive.vgpr6, i32 %inactive.vgpr7,
                                    i32 %inactive.vgpr8, i32 %inactive.vgpr9)
                                    local_unnamed_addr {
entry:
  %system.data.value = extractvalue { i32 } %system.data, 0
  %dead.val = call i32 @llvm.amdgcn.dead.i32()
  %is.whole.wave = call i1 @llvm.amdgcn.init.whole.wave()
  br i1 %is.whole.wave, label %shader, label %tail

shader:
  %system.data.extract = extractvalue { i32 } %system.data, 0
  %data.mul = mul i32 %system.data.extract, 2
  %data.add = add i32 %data.mul, 1
  call void asm sideeffect "; clobber VGPR for %inactive.vgpr2", "~{v12}"()
  br label %tail

tail:
  %final.vcr = phi i32 [ %vcr, %entry ], [ %data.mul, %shader ]
  %final.sys.data = phi i32 [ %system.data.value, %entry ], [ %data.add, %shader ]
  %final.inactive0 = phi i32 [ %inactive.vgpr, %entry ], [ %dead.val, %shader ]
  %final.inactive1 = phi i32 [ %inactive.vgpr1, %entry ], [ %dead.val, %shader ]
  %final.inactive2 = phi i32 [ %inactive.vgpr2, %entry ], [ %dead.val, %shader ]
  %final.inactive3 = phi i32 [ %inactive.vgpr3, %entry ], [ %dead.val, %shader ]
  %final.inactive4 = phi i32 [ %inactive.vgpr4, %entry ], [ %dead.val, %shader ]
  %final.inactive5 = phi i32 [ %inactive.vgpr5, %entry ], [ %dead.val, %shader ]
  %final.inactive6 = phi i32 [ %inactive.vgpr6, %entry ], [ %dead.val, %shader ]
  %final.inactive7 = phi i32 [ %inactive.vgpr7, %entry ], [ %dead.val, %shader ]
  %final.inactive8 = phi i32 [ %inactive.vgpr8, %entry ], [ %dead.val, %shader ]
  %final.inactive9 = phi i32 [ %inactive.vgpr9, %entry ], [ %dead.val, %shader ]

  %struct.init = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } poison, i32 %final.vcr, 0
  %struct.with.data = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.init, i32 %final.sys.data, 1
  %struct.with.inactive0 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.data, i32 %final.inactive0, 2
  %struct.with.inactive1 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.inactive0, i32 %final.inactive1, 3
  %struct.with.inactive2 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.inactive1, i32 %final.inactive2, 4
  %struct.with.inactive3 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.inactive2, i32 %final.inactive3, 5
  %struct.with.inactive4 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.inactive3, i32 %final.inactive4, 6
  %struct.with.inactive5 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.inactive4, i32 %final.inactive5, 7
  %struct.with.inactive6 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.inactive5, i32 %final.inactive6, 8
  %struct.with.inactive7 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.inactive6, i32 %final.inactive7, 9
  %struct.with.inactive8 = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.inactive7, i32 %final.inactive8, 10
  %final.struct = insertvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %struct.with.inactive8, i32 %final.inactive9, 11

  %vec.global = insertelement <4 x i32> poison, i32 %global.table, i64 0
  %vec.max.vgpr = insertelement <4 x i32> %vec.global, i32 %max.outgoing.vgpr.count, i64 1
  %vec.sys.data = insertelement <4 x i32> %vec.max.vgpr, i32 %final.sys.data, i64 2
  %final.vec = insertelement <4 x i32> %vec.sys.data, i32 0, i64 3

  call void (ptr, i32, <4 x i32>, { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, i32, ...)
        @llvm.amdgcn.cs.chain.p0.i32.v4i32.sl_i32i32i32i32i32i32i32i32i32i32i32i32s(
        ptr %next.callee, i32 0, <4 x i32> inreg %final.vec,
        { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %final.struct,
        i32 1, i32 %max.outgoing.vgpr.count, i32 -1, ptr @retry_vgpr_alloc.v4i32)
  unreachable
}

declare i32 @llvm.amdgcn.dead.i32()
declare i1 @llvm.amdgcn.init.whole.wave()
declare void @llvm.amdgcn.cs.chain.p0.i32.v4i32.sl_i32i32i32i32i32i32i32i32i32i32i32i32s(ptr, i32, <4 x i32>, { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, i32 immarg, ...)

declare amdgpu_cs_chain void @retry_vgpr_alloc.v4i32(<4 x i32> inreg)
