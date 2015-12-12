; RUN: sed -e s/.T1:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK1 %s
; RUN: sed -e s/.T2:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK2 %s
; RUN: sed -e s/.T3:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK3 %s
; RUN: sed -e s/.T4:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK4 %s

;T1: define void @f() {
;T1:   entry:
;T1:     catchret from undef to label %next
;T1:     ; CHECK1: CatchReturnInst needs to be provided a CatchPad
;T1:   next:
;T1:     unreachable
;T1: }

;T2: define void @f() {
;T2:   entry:
;T2:     %x = cleanuppad within none []
;T2:     ; catchret's first operand's operator must be catchpad
;T2:     catchret from %x to label %entry
;T2:     ; CHECK2: CatchReturnInst needs to be provided a CatchPad
;T2: }

;T3: define void @f() {
;T3:   entry:
;T3:     cleanupret from undef unwind label %next
;T3:     ; CHECK3: CleanupReturnInst needs to be provided a CleanupPad
;T3:   next:
;T3:     unreachable
;T3: }

;T4: define void @f() {
;T4:   entry:
;T4:     %cs = catchswitch within none [label %next] unwind to caller
;T4:   next:
;T4:     %x = catchpad within %cs []
;T4:     ; cleanupret first operand's operator must be cleanuppad
;T4:     cleanupret from %x unwind to caller
;T4:     ; CHECK4: CleanupReturnInst needs to be provided a CleanupPad
;T4: }
