; RUN: llc -march=mips -mcpu=mips32 -verify-machineinstrs -o /dev/null < %s 
; Confirm that MachineInstr branch simplification preserves
; register operand flags, such as the <undef> flag.

define void @ham() {
bb:
  %tmp = alloca i32, align 4
  %tmp13 = ptrtoint ptr %tmp to i32
  %tmp70 = icmp eq i32 undef, -1
  br i1 %tmp70, label %bb72, label %bb40

bb72:                                             ; preds = %bb72, %bb
  br i1 undef, label %bb40, label %bb72

bb40:                                             ; preds = %bb72, %bb
  %tmp41 = phi i32 [ %tmp13, %bb72 ], [ %tmp13, %bb ]
  %tmp55 = inttoptr i32 %tmp41 to ptr
  %tmp58 = insertelement <2 x ptr> undef, ptr %tmp55, i32 1
  br label %bb59

bb59:                                             ; preds = %bb59, %bb40
  %tmp60 = phi <2 x ptr> [ %tmp61, %bb59 ], [ %tmp58, %bb40 ]
  %tmp61 = getelementptr i32, <2 x ptr> %tmp60, <2 x i32> <i32 -1, i32 1>
  %tmp62 = extractelement <2 x ptr> %tmp61, i32 1
  br label %bb59
}
