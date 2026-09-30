; RUN: llc -mtriple=amdgpu9.00 -stop-after=finalize-isel -o - %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.00 -global-isel -stop-after=instruction-select -o - %s | FileCheck %s

; SI_END_CF at the top of a join block is a block prolog instruction, ahead of
; the block's debug instructions.

; CHECK-LABEL: name: end_cf_dbg_value
; CHECK: bb.{{[0-9]+}}.exit:
; CHECK-NOT: {{^  bb\.}}
; CHECK: bb-prolog SI_END_CF
; CHECK-NEXT: DBG_VALUE

; GlobalISel copies the saved mask in front of the end.cf intrinsic; the copy
; is gone after selection and SI_END_CF still starts the join block.

; CHECK-LABEL: name: end_cf_after_mask_copy
; CHECK: bb.{{[0-9]+}}.endif:
; CHECK-NOT: {{^  bb\.}}
; CHECK: bb-prolog SI_END_CF

define void @end_cf_dbg_value(ptr addrspace(1) %out) !dbg !4 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %acc.next, %loop ]
  %f = sitofp i32 %i to float
  %acc.next = fadd float %acc, %f
  %i.next = add i32 %i, 1
  %cmp = icmp uge i32 %i.next, %tid
  br i1 %cmp, label %exit, label %loop

exit:
  %acc.lcssa = phi float [ %acc.next, %loop ]
    #dbg_value(float %acc.lcssa, !6, !DIExpression(), !8)
  %gep = getelementptr float, ptr addrspace(1) %out, i32 %tid
  store float %acc.lcssa, ptr addrspace(1) %gep
  ret void
}

define amdgpu_kernel void @end_cf_after_mask_copy(ptr addrspace(1) %out, i32 %a) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp eq i32 %tid, 0
  br i1 %cmp, label %if, label %endif

if:
  %v = add i32 %a, %tid
  br label %endif

endif:
  %phi = phi i32 [ %v, %if ], [ 0, %entry ]
  store i32 %phi, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "end_cf_dbg_value", scope: !1, file: !1, line: 1, type: !5, unit: !0, spFlags: DISPFlagDefinition)
!5 = !DISubroutineType(types: !{})
!6 = !DILocalVariable(name: "acc", scope: !4, file: !1, line: 1, type: !7)
!7 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!8 = !DILocation(line: 1, scope: !4)
