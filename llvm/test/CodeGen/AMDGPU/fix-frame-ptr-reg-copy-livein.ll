; RUN: llc -march=amdgcn -mcpu=gfx900 -stop-after=prologepilog -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; It is a small loop test that iterates over the array member of the structure argument  passed byval to the function.
; The loop code will keep the prologue and epilogue blocks apart.
; The test is primarily to check the temp register used to preserve the earlier FP value
; is live-in at every BB in the function.

%struct.Data = type { [20 x i32] }

define i32 @fp_save_restore_in_temp_sgpr(%struct.Data addrspace(5)* nocapture readonly byval(%struct.Data) align 4 %arg) #0 {
  ; GCN-LABEL: name: fp_save_restore_in_temp_sgpr
  ; GCN: bb.0.begin:
  ; GCN:   liveins: $sgpr11
  ; GCN:   $sgpr11 = frame-setup COPY $sgpr33
  ; GCN:   $sgpr33 = frame-setup COPY $sgpr32
  ; GCN: bb.1.lp_end:
  ; GCN:   liveins: $sgpr10, $sgpr11, $vgpr1, $sgpr4_sgpr5, $sgpr6_sgpr7, $sgpr8_sgpr9
  ; GCN: bb.2.lp_begin:
  ; GCN:   liveins: $sgpr10, $sgpr11, $vgpr1, $sgpr4_sgpr5, $sgpr6_sgpr7
  ; GCN: bb.3.Flow:
  ; GCN:   liveins: $sgpr10, $sgpr11, $vgpr0, $vgpr1, $sgpr4_sgpr5, $sgpr6_sgpr7, $sgpr8_sgpr9
  ; GCN: bb.4.end:
  ; GCN:   liveins: $sgpr11, $vgpr0, $sgpr4_sgpr5
  ; GCN:   $sgpr33 = frame-destroy COPY $sgpr11
begin:
  br label %lp_begin

lp_end:                                                ; preds = %lp_begin
  %cur_idx = add nuw nsw i32 %idx, 1
  %lp_term_cond = icmp eq i32 %cur_idx, 20
  br i1 %lp_term_cond, label %end, label %lp_begin

lp_begin:                                                ; preds = %lp_end, %begin
  %idx = phi i32 [ 0, %begin ], [ %cur_idx, %lp_end ]
  %ptr = getelementptr inbounds %struct.Data, %struct.Data addrspace(5)* %arg, i32 0, i32 0, i32 %idx
  %data = load i32, i32 addrspace(5)* %ptr, align 4
  %data_cmp = icmp eq i32 %data, %idx
  br i1 %data_cmp, label %lp_end, label %end

end:                                               ; preds = %lp_end, %lp_begin
  %ret_val = phi i32 [ 0, %lp_begin ], [ 1, %lp_end ]
  ret i32 %ret_val
}

attributes #0 = { norecurse nounwind "frame-pointer"="all" }
