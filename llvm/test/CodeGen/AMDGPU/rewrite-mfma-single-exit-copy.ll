; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa -amdgpu-disable-rewrite-mfma-form-sched-stage=false < %s | FileCheck %s

; Test that the MFMA rewrite stage generates only one AGPR->VGPR copy
; when multiple VGPR-requiring instructions in the exit block use the
; same MFMA result.
;
; 8 x <32 x float> loop-carried carriers create 256 VGPRs of pressure.
; 3 chained MFMAs in the loop, then in the exit block 1 MFMA (m3)
; followed by 2 MFMAs (out1, out2) that both use m3 as src2, plus
; fadd uses of out1/out2 that require VGPRs.
; The two exit MFMAs should share one v_accvgpr_read copy of m3.

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

define amdgpu_kernel void @single_exit_copy(
; CHECK-LABEL: single_exit_copy:
; CHECK-NOT: v_accvgpr_read_b32
; CHECK: ; %bb.2: ; %exit
; CHECK-NOT: v_accvgpr_read_b32
; CHECK: v_mfma_f32_16x16x32_f16 a[0:3], a[4:7], a[8:11], a[0:3]
; CHECK-NOT: v_accvgpr_read_b32
; CHECK: v_accvgpr_read_b32 v[[#HI:]], a3
; CHECK-NEXT: v_accvgpr_read_b32 v[[#HI - 1]], a2
; CHECK-NEXT: v_accvgpr_read_b32 v[[#HI - 2]], a1
; CHECK-NEXT: v_accvgpr_read_b32 v[[#HI - 3]], a0
; CHECK-NOT: v_accvgpr_read_b32
; CHECK: v_mfma_f32_16x16x32_f16 v{{\[[0-9]+:[0-9]+\]}}, a[4:7], a[8:11], v{{\[}}[[#HI - 3]]:[[#HI]]{{\]}}
; CHECK-NOT: v_accvgpr_read_b32
; CHECK: v_mfma_f32_16x16x32_f16 v{{\[[0-9]+:[0-9]+\]}}, a[8:11], a[4:7], v{{\[}}[[#HI - 3]]:[[#HI]]{{\]}}
; CHECK-NOT: v_accvgpr_read_b32
; CHECK: s_endpgm
; Loop body: MFMAs in AGPR form.
; Exit block: one MFMA consuming the loop result (AGPR form), then
; exactly 4 v_accvgpr_read (one 128-bit copy), shared by both vgprcd MFMAs.

    <8 x half> %a, <8 x half> %b, i32 %n, ptr addrspace(1) %out
) #0 !dbg !4 {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi <4 x float> [ zeroinitializer, %entry ], [ %m2, %loop ]
  ; 8 loop-carried <32 x float> vectors = 256 VGPRs of pressure.
  %vc0 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc0n, %loop ]
  %vc1 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc1n, %loop ]
  %vc2 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc2n, %loop ]
  %vc3 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc3n, %loop ]
  %vc4 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc4n, %loop ]
  %vc5 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc5n, %loop ]
  %vc6 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc6n, %loop ]
  %vc7 = phi <32 x float> [ zeroinitializer, %entry ], [ %vc7n, %loop ]

  %m0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a, <8 x half> %b, <4 x float> %acc, i32 0, i32 0, i32 0)
  %m1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a, <8 x half> %b, <4 x float> %m0, i32 0, i32 0, i32 0)
  %m2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a, <8 x half> %b, <4 x float> %m1, i32 0, i32 0, i32 0)

  ; Keep carriers alive.
  %vc0n = fadd <32 x float> %vc0, %vc0
  %vc1n = fadd <32 x float> %vc1, %vc1
  %vc2n = fadd <32 x float> %vc2, %vc2
  %vc3n = fadd <32 x float> %vc3, %vc3
  %vc4n = fadd <32 x float> %vc4, %vc4
  %vc5n = fadd <32 x float> %vc5, %vc5
  %vc6n = fadd <32 x float> %vc6, %vc6
  %vc7n = fadd <32 x float> %vc7, %vc7

  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  %m3 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a, <8 x half> %b, <4 x float> %m2, i32 0, i32 0, i32 0), !dbg !7
  %out1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %a, <8 x half> %b, <4 x float> %m3, i32 0, i32 0, i32 0), !dbg !8
  %out2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %b, <8 x half> %a, <4 x float> %m3, i32 0, i32 0, i32 0), !dbg !9

  ; VGPR-requiring uses of the exit MFMAs to force accvgpr_read copies.
  %e0 = extractelement <4 x float> %out1, i32 0
  %e1 = extractelement <4 x float> %out2, i32 0
  %add0 = fadd float %e0, 1.0
  %add1 = fadd float %e1, 2.0

  ; Store results + carriers to keep everything live.
  store float %add0, ptr addrspace(1) %out, align 4
  %gep1 = getelementptr float, ptr addrspace(1) %out, i32 1
  store float %add1, ptr addrspace(1) %gep1, align 4
  %gep2 = getelementptr <32 x float>, ptr addrspace(1) %out, i32 15
  store <32 x float> %vc0n, ptr addrspace(1) %gep2, align 128
  %gep3 = getelementptr <32 x float>, ptr addrspace(1) %out, i32 2
  store <32 x float> %vc1n, ptr addrspace(1) %gep3, align 128
  %gep4 = getelementptr <32 x float>, ptr addrspace(1) %out, i32 3
  store <32 x float> %vc2n, ptr addrspace(1) %gep4, align 128
  %gep5 = getelementptr <32 x float>, ptr addrspace(1) %out, i32 4
  store <32 x float> %vc3n, ptr addrspace(1) %gep5, align 128
  %gep6 = getelementptr <32 x float>, ptr addrspace(1) %out, i32 5
  store <32 x float> %vc4n, ptr addrspace(1) %gep6, align 128
  %gep7 = getelementptr <32 x float>, ptr addrspace(1) %out, i32 6
  store <32 x float> %vc5n, ptr addrspace(1) %gep7, align 128
  %gep8 = getelementptr <32 x float>, ptr addrspace(1) %out, i32 7
  store <32 x float> %vc6n, ptr addrspace(1) %gep8, align 128
  %gep9 = getelementptr <32 x float>, ptr addrspace(1) %out, i32 8
  store <32 x float> %vc7n, ptr addrspace(1) %gep9, align 128
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="64,64" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "single_exit_copy", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocation(line: 10, column: 1, scope: !4)
!8 = !DILocation(line: 20, column: 1, scope: !4)
!9 = !DILocation(line: 30, column: 1, scope: !4)
