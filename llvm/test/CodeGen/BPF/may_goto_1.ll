; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o - %s | llvm-objdump --no-show-raw-insn -d - | FileCheck %s

@j = dso_local local_unnamed_addr global i32 0, align 4

define dso_local noundef i32 @foo() local_unnamed_addr {
entry:
  callbr void asm sideeffect "may_goto ${0:l}", "!i"()
          to label %for.body [label %for.cond.cleanup]

for.cond.cleanup:                                 ; preds = %for.body.2, %for.body.2, %for.body.1, %for.body, %entry
  ret i32 0

for.body:                                         ; preds = %entry
  callbr void asm sideeffect "may_goto ${0:l}", "!i"()
          to label %for.body.1 [label %for.cond.cleanup]

for.body.1:                                       ; preds = %for.body
  callbr void asm sideeffect "may_goto ${0:l}", "!i"()
          to label %for.body.2 [label %for.cond.cleanup]

for.body.2:                                       ; preds = %for.body.1
  callbr void asm sideeffect "may_goto ${0:l}", "!i"()
          to label %for.cond.cleanup [label %for.cond.cleanup]
}

; CHECK:       0:       w0 = 0x0
; CHECK-NEXT:  1:       exit
