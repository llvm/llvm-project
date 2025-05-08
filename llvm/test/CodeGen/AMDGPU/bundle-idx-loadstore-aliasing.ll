; RUN: llc -march=amdgcn -mcpu=gfx1300 -verify-machineinstrs -debug-only=bundle-indexed-load-store -stop-after=bundle-indexed-load-store < %s 2> %t | FileCheck --check-prefixes=SINGLEBB %s
; RUN: FileCheck --check-prefixes=SINGLEBBDBG %s < %t
; RUN: FileCheck --check-prefixes=MULTIBBDBG %s < %t

; The two tests in this file demonstrate alias analysis (AA) working in two different contexts for the 
; AMDGPUBundleIdxLdSt pass: the single basic block use case, and the multi basic block use case. In both
; test cases, both of the AA outcomes (presence and absence of an alias conflict) are shown. Because the 
; single basic block test case's CFG is simple enough, the MIR after the pass is also included.

target triple = "amdgcn-amd-amdhsa"
@weights = external local_unnamed_addr addrspace(10) global [256 x i32], align 4
@out = external local_unnamed_addr addrspace(10) global [32 x i32], align 4

define dso_local amdgpu_kernel void @amdgcn_aa_singlebb() local_unnamed_addr {
; SINGLEBBDBG:   ===== AMDGPUBundleIdxLdSt :: Bundling Phase =====
; SINGLEBBDBG-NEXT: BB.0 :: [[S_MOV_B32_:%[0-9]+]]:sgpr_32 = S_MOV_B32 0
; SINGLEBBDBG-NEXT: BB.0 :: [[V_LOAD_IDX_:%[0-9]+]]:vgpr_32 = V_LOAD_IDX [[S_MOV_B32_]]:sgpr_32, 82, implicit $exec :: (dereferenceable load (s32) from `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, addrspace 10)
; SINGLEBBDBG-NEXT: BB.0 :: [[V_MOV_B32_:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 5, implicit $exec
; SINGLEBBDBG-NEXT:  *** Created bundle from
; SINGLEBBDBG-NEXT:         $stg_dsta = V_MOV_B32_e32 5, implicit $exec
; SINGLEBBDBG-NEXT:         V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]]:sgpr_32, 82, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, addrspace 10)
; SINGLEBBDBG-NEXT: BB.0 :: V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]]:sgpr_32, 82, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, addrspace 10)
; SINGLEBBDBG-NEXT: BB.0 :: [[V_LOAD_IDX_1:%[0-9]+]]:vgpr_32 = V_LOAD_IDX [[S_MOV_B32_]]:sgpr_32, 87, implicit $exec :: (dereferenceable load (s32) from `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 220)`, addrspace 10)
; SINGLEBBDBG-NEXT: BB.0 :: [[V_MOV_B32:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 7, implicit $exec
; SINGLEBBDBG-NEXT:  *** Created bundle from
; SINGLEBBDBG-NEXT:         $stg_dsta = V_MOV_B32_e32 7, implicit $exec
; SINGLEBBDBG-NEXT:         V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]]:sgpr_32, 80, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 192)`, addrspace 10)
; SINGLEBBDBG-NEXT: BB.0 :: V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]]:sgpr_32, 80, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 192)`, addrspace 10)
; SINGLEBBDBG-NEXT: BB.0 :: [[V_ADD_U32:%[0-9]+]]:vgpr_32 = nsw V_ADD_U32_e64 killed [[V_LOAD_IDX_1]]:vgpr_32, killed [[V_LOAD_IDX_]]:vgpr_32, 0, implicit $exec
; SINGLEBBDBG-NEXT:  *** Conflict with V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]]:sgpr_32, 82, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, addrspace 10)
; SINGLEBBDBG-NEXT:  *** Created bundle from
; SINGLEBBDBG-NEXT:         $stg_srca = V_LOAD_IDX [[S_MOV_B32_]]:sgpr_32, 87, implicit $exec :: (dereferenceable load (s32) from `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 220)`, addrspace 10)
; SINGLEBBDBG-NEXT:         $stg_dsta = nsw V_ADD_U32_e64 internal killed $stg_srca, killed [[V_LOAD_IDX_]]:vgpr_32, 0, implicit $exec
; SINGLEBBDBG-NEXT:         V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]]:sgpr_32, 16, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @out, i32 64)`, addrspace 10)
; SINGLEBBDBG-NEXT: BB.0 :: V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]]:sgpr_32, 16, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @out, i32 64)`, addrspace 10)
; SINGLEBBDBG-NEXT: BB.0 :: S_ENDPGM 0
; SINGLEBB-LABEL: name:            amdgcn_aa_singlebb
; SINGLEBB: [[S_MOV_B32_:%[0-9]+]]:sgpr_32 = S_MOV_B32 0
; SINGLEBB-NEXT:    [[V_LOAD_IDX_:%[0-9]+]]:vgpr_32 = V_LOAD_IDX [[S_MOV_B32_]], 82, implicit $exec :: (dereferenceable load (s32) from `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, addrspace 10)
; SINGLEBB-NEXT:    BUNDLE implicit-def $stg_dsta, implicit $exec, implicit [[S_MOV_B32_]] {
; SINGLEBB-NEXT:      $stg_dsta = V_MOV_B32_e32 5, implicit $exec
; SINGLEBB-NEXT:      V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]], 82, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, addrspace 10)
; SINGLEBB-NEXT:    }
; SINGLEBB-NEXT:    BUNDLE implicit-def $stg_dsta, implicit $exec, implicit [[S_MOV_B32_]] {
; SINGLEBB-NEXT:      $stg_dsta = V_MOV_B32_e32 7, implicit $exec
; SINGLEBB-NEXT:      V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]], 80, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 192)`, addrspace 10)
; SINGLEBB-NEXT:    }
; SINGLEBB-NEXT:    BUNDLE implicit-def dead $stg_srca, implicit-def $stg_dsta, implicit [[S_MOV_B32_]], implicit $exec, implicit killed [[V_LOAD_IDX_]] {
; SINGLEBB-NEXT:      $stg_srca = V_LOAD_IDX [[S_MOV_B32_]], 87, implicit $exec :: (dereferenceable load (s32) from `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 220)`, addrspace 10)
; SINGLEBB-NEXT:      $stg_dsta = nsw V_ADD_U32_e64 internal killed $stg_srca, killed [[V_LOAD_IDX_]], 0, implicit $exec
; SINGLEBB-NEXT:      V_STORE_IDX internal $stg_dsta, [[S_MOV_B32_]], 16, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @out, i32 64)`, addrspace 10)
; SINGLEBB-NEXT:    }
; SINGLEBB-NEXT:    S_ENDPGM 0
entry:
  %0 = load i32, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200), align 4
  store i32 5, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200), align 4
  %1 = load i32, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 220), align 4
  store i32 7, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 192), align 4
  %add3 = add nsw i32 %1, %0
  store i32 %add3, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @out, i32 64), align 4
  ret void
}

define dso_local amdgpu_kernel void @amdgcn_aa_multibb() local_unnamed_addr {
; MULTIBBDBG: ===== AMDGPUBundleIdxLdSt :: Sinking Phase =====
;     Skip first kernel.
; MULTIBBDBG: ===== AMDGPUBundleIdxLdSt :: Sinking Phase =====
; MULTIBBDBG:  *** Conflict with V_STORE_IDX [[V_STORE_IDX_:%[0-9]+]]:vgpr_32, %39:sgpr_32, 82, implicit $exec :: (store (s32) into `ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200)`, addrspace 10)
; MULTIBBDBG: BB.2 :: [[V_STORE_IDX_]]:vgpr_32 = V_MOV_B32_e32 5, implicit $exec
; MULTIBBDBG-NEXT:  *** Found 1 use(s)
; MULTIBBDBG-NEXT:  *** Use is in MI's current block. Leaving a copy in block 2
; MULTIBBDBG: BB.4 :: [[V_LOAD_IDX_:%[0-9]+]]:vgpr_32 = V_LOAD_IDX [[V_LOAD_IDX_1:%[0-9]+]]:sreg_32_xm0, 82, implicit $exec :: (load (s32) from %ir.arrayidx2, addrspace 10)
; MULTIBBDBG-NEXT:  *** Found 1 use(s)
; MULTIBBDBG-NEXT:  *** Sinking MI to block [[BLOCK_:[0-9]+]]
; MULTIBBDBG: BB.5 :: [[V_ADD_:%[0-9]+]]:vgpr_32 = nsw V_ADD_U32_e64 killed [[V_LOAD_IDX_]]:vgpr_32, killed [[_:%[0-9]+]]:vgpr_32, 0, implicit $exec
; MULTIBBDBG-NEXT:  *** CoreMI sinking to larger cycle depth is not profitable
; MULTIBBDBG: BB.5 :: [[_:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 7, implicit $exec
; MULTIBBDBG-NEXT:  *** Found 1 use(s)
; MULTIBBDBG-NEXT:  *** Use is in MI's current block. Leaving a copy in block [[BLOCK_]]
; MULTIBBDBG: BB.5 :: [[V_LOAD_IDX_]]:vgpr_32 = V_LOAD_IDX [[V_LOAD_IDX_1]]:sreg_32_xm0, 82, implicit $exec :: (load (s32) from %ir.arrayidx2, addrspace 10)
; MULTIBBDBG-NEXT:  *** Found 1 use(s)
; MULTIBBDBG-NEXT:  *** Use is in MI's current block. Leaving a copy in block [[BLOCK_]]
; MULTIBBDBG: BB.7 :: V_STORE_IDX [[V_ADD_]]:vgpr_32, [[_:%[0-9]+]]:sreg_32_xm0, 0, implicit $exec :: (store (s32) into %ir.arrayidx6, addrspace 10)
entry:
  %0 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %add = add nuw nsw i32 %0, 50
  %arrayidx = getelementptr inbounds nuw [256 x i32], ptr addrspace(10) @weights, i32 0, i32 %add
  %1 = load i32, ptr addrspace(10) %arrayidx, align 4
  store i32 5, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 200), align 4
  %2 = load i32, ptr addrspace(10) %arrayidx, align 4
  store i32 7, ptr addrspace(10) getelementptr inbounds nuw (i8, ptr addrspace(10) @weights, i32 192), align 4
  %add4 = add nsw i32 %2, %1
  %arrayidx6 = getelementptr inbounds nuw [32 x i32], ptr addrspace(10) @out, i32 0, i32 %0
  store i32 %add4, ptr addrspace(10) %arrayidx6, align 4
  ret void
}
