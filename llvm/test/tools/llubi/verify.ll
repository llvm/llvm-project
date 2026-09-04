; RUN: not llubi < %s 2>&1 | FileCheck %s --check-prefix=VERIFY
; RUN: llubi --disable-verify --verbose < %s 2>&1 | FileCheck %s --check-prefix=NO-VERIFY

define i32 @main() {
entry:
  br label %next

next:
  %x = phi i32 [ 0, %entry ], [ 1, %entry ]
  ret i32 %x
}

; VERIFY: PHINode should have one entry for each predecessor of its parent basic block!
; VERIFY: error: {{.*}}: input module is broken!

; NO-VERIFY-NOT: input module is broken!
; NO-VERIFY: Entering function: main
; NO-VERIFY: ret i32 %x
; NO-VERIFY: Exiting function: main
; NO-VERIFY-NOT: input module is broken!
