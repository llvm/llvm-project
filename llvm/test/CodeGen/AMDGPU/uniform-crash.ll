; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}icmp_2_users:
; GCN: s_cmp_lt_i32 s{{[0-9]+}}, 1
; GCN: s_cbranch_scc1 [[LABEL:BB[0-9_A-Z]+]]
; GCN: [[LABEL]]:
; GCN-NEXT: s_endpgm
define void @icmp_2_users(i32 addrspace(1)* %out, i32 %cond) {
main_body:
  %0 = icmp sgt i32 %cond, 0
  %1 = sext i1 %0 to i32
  br i1 %0, label %IF, label %ENDIF

IF:
  store i32 %1, i32 addrspace(1)* %out
  br label %ENDIF

ENDIF:                                            ; preds = %IF, %main_body
  ret void
}

; GCN-LABEL: {{^}}fix_sgpr_live_ranges_crash:
; GCN: s_cbranch_scc1 [[BB0:[A-Z0-9_]+]]
; GCN: {{^}}[[LOOP:[A-Z0-9_]+]]:
; GCN: s_cbranch_scc1 [[LOOP]]
; GCN: {{^}}[[BB0]]:
define void @fix_sgpr_live_ranges_crash(i32 %arg, i32 %arg1)  {
bb:
  %cnd = trunc i32 %arg to i1
  br i1 %cnd, label %bb2, label %bb5

bb2:                                              ; preds = %bb
  %tmp = mul i32 10, %arg1
  br label %bb3

bb3:                                              ; preds = %bb3, %bb2
  %val = load volatile i32, i32 addrspace(2)* undef
  %tmp4 = icmp eq i32 %val, %arg1
  br i1 %tmp4, label %bb5, label %bb3

bb5:                                              ; preds = %bb3, %bb
  %tmp6 = tail call i32 @llvm.amdgcn.workitem.id.y() #1
  %tmp10 = icmp ult i32 %tmp6, %arg
  br i1 %tmp10, label %bb11, label %bb12

bb11:                                             ; preds = %bb11, %bb5
  br i1 undef, label %bb11, label %bb12

bb12:                                             ; preds = %bb11, %bb5
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() #1

attributes #1 = { nounwind readnone }
