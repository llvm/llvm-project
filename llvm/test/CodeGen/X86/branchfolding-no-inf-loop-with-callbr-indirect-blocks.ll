; RUN: llc -mtriple x86_64 %s -stop-after=branch-folder

; Check branch-folder do not keep swapping %indirect.target.block.1 and
; %indirect.target.block.2
define void @no_inf_loop() {
entry:
  br label %bb1

bb1:
  callbr void asm "", "!i,!i"() to label %bb1
    [label %indirect.target.block.1, label %indirect.target.block.2]

indirect.target.block.1:
  br label %bb1

indirect.target.block.2:
  br label %bb1
}
