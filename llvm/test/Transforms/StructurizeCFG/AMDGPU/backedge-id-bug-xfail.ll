; XFAIL: *
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -structurizecfg -verify-region-info %s -enable-new-pm=0

; FIXME: Merge into backedge-id-bug
; Variant which has an issue with region construction

define amdgpu_kernel void @loop_backedge_misidentified_alt(i32 addrspace(1)* %arg0) #0 {
entry:
  %tmp = load volatile <2 x i32>, <2 x i32> addrspace(1)* undef, align 16
  %load1 = load volatile <2 x float>, <2 x float> addrspace(1)* undef
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i32 %tid
  %i.initial = load volatile i32, i32 addrspace(1)* %gep, align 4
  br label %LOOP.HEADER

LOOP.HEADER:
  %i = phi i32 [ %i.final, %END_ELSE_BLOCK ], [ %i.initial, %entry ]
  call void asm sideeffect "s_nop 0x100b ; loop $0 ", "r,~{memory}"(i32 %i) #0
  %tmp12 = zext i32 %i to i64
  %tmp13 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* null, i64 %tmp12
  %tmp14 = load <4 x i32>, <4 x i32> addrspace(1)* %tmp13, align 16
  %tmp15 = extractelement <4 x i32> %tmp14, i64 0
  %tmp16 = and i32 %tmp15, 65535
  %tmp17 = icmp eq i32 %tmp16, 1
  br i1 %tmp17, label %bb18, label %bb62

bb18:
  %tmp19 = extractelement <2 x i32> %tmp, i64 0
  %tmp22 = lshr i32 %tmp19, 16
  %tmp24 = urem i32 %tmp22, 52
  %tmp25 = mul nuw nsw i32 %tmp24, 52
  br label %INNER_LOOP

INNER_LOOP:
  %inner.loop.j = phi i32 [ %tmp25, %bb18 ], [ %inner.loop.j.inc, %INNER_LOOP ]
  call void asm sideeffect "; inner loop body", ""() #0
  %inner.loop.j.inc = add nsw i32 %inner.loop.j, 1
  %inner.loop.cmp = icmp eq i32 %inner.loop.j, 0
  br i1 %inner.loop.cmp, label %INNER_LOOP_BREAK, label %INNER_LOOP

INNER_LOOP_BREAK:
  %tmp59 = extractelement <4 x i32> %tmp14, i64 2
  call void asm sideeffect "s_nop 23 ", "~{memory}"() #0
  br label %END_ELSE_BLOCK

bb62:
  %load13 = icmp ult i32 %tmp16, 271
  ;br i1 %load13, label %bb64, label %INCREMENT_I
  ; branching directly to the return avoids the bug
  br i1 %load13, label %RETURN, label %INCREMENT_I


bb64:
  call void asm sideeffect "s_nop 42", "~{memory}"() #0
  br label %RETURN

INCREMENT_I:
  %inc.i = add i32 %i, 1
  call void asm sideeffect "s_nop 0x1336 ; increment $0", "v,~{memory}"(i32 %inc.i) #0
  br label %END_ELSE_BLOCK

END_ELSE_BLOCK:
  %i.final = phi i32 [ %tmp59, %INNER_LOOP_BREAK ], [ %inc.i, %INCREMENT_I ]
  call void asm sideeffect "s_nop 0x1337 ; end else block $0", "v,~{memory}"(i32 %i.final) #0
  %cmp.end.else.block = icmp eq i32 %i.final, -1
  br i1 %cmp.end.else.block, label %RETURN, label %LOOP.HEADER

RETURN:
  call void asm sideeffect "s_nop 0x99 ; ClosureEval return", "~{memory}"() #0
  store volatile <2 x float> %load1, <2 x float> addrspace(1)* undef, align 8
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nounwind readnone }
