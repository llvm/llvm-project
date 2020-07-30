; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -verify-machineinstrs -stop-after=amdgpu-isel -o - %s | FileCheck -check-prefix=GCN %s
define void @test() #1 {
  ; Clean up the unreachable blocks introduced with LowerSwitch pass.
  ; This test ensures that, in the pass flow, UnreachableBlockElim pass
  ; follows the LowerSwitch. Otherwise, this testcase will crash
  ; immediately after the instruction selection due to the incomplete
  ; PHI node in an MBB whose incoming values were never codegenerated.
  ;
  ; GCN-LABEL: name: test
  ; GCN: bb.{{[0-9]+}}.entry:
  ; GCN: bb.{{[0-9]+}}.entry.true.blk:
  ; GCN: bb.{{[0-9]+}}.entry.false.blk:
  ; GCN: bb.{{[0-9]+}}.switch.blk:

  ; GCN-NOT: bb.{{[0-9]+}}.preheader.blk
  ; GCN-NOT: bb.{{[0-9]+}}.pre.false.blk:
  ; GCN-NOT: bb.{{[0-9]+}}.unreach.blk:
  ; GCN-NOT: PHI

  ; GCN: bb.{{[0-9]+}}.exit:
  entry:
    %idx = tail call i32 @llvm.amdgcn.workitem.id.x() #0
    br i1 undef, label %entry.true.blk, label %entry.false.blk

  entry.true.blk:                                   ; preds = %entry
    %exit.cmp = icmp ult i32 %idx, 3
    br i1 %exit.cmp, label %switch.blk, label %exit

  entry.false.blk:                                  ; preds = %entry
    unreachable

  switch.blk:                                       ; preds = %entry.true.blk
    switch i32 %idx, label %preheader.blk [
      i32 0, label %exit
      i32 1, label %exit
      i32 2, label %exit
    ]

  preheader.blk:                                    ; preds = %switch.blk
    %pre.exit = icmp ult i32 %idx, 5
    br i1 %pre.exit, label %unreach.blk, label %pre.false.blk

  pre.false.blk:                                    ; preds = %preheader.blk
    %call.pre.false = tail call i32 @func(i32 %idx) #0
    br label %unreach.blk

  unreach.blk:                                      ; preds = %preheader.blk, %pre.false.blk
    %phi.val = phi i32 [ %call.pre.false, %pre.false.blk ], [ undef, %preheader.blk ]
    store i32 %phi.val, i32* undef
    unreachable

  exit:                                             ; preds = %switch.blk
    ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @func(i32)#0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
