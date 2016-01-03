; RUN: sed -e s/.T1:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK1 %s
; RUN: sed -e s/.T2:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK2 %s
; RUN: sed -e s/.T3:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK3 %s
; RUN: sed -e s/.T4:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK4 %s
; RUN: sed -e s/.T5:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK5 %s
; RUN: sed -e s/.T6:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK6 %s
; RUN: sed -e s/.T7:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK7 %s
; RUN: sed -e s/.T8:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK8 %s

declare void @g()

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

;T5: define void @f() personality void ()* @g {
;T5:   entry:
;T5:     ret void
;T5:   switch:
;T5:     %cs = catchswitch within none [label %catch] unwind to caller
;T5:   catch:
;T5:     catchpad within %cs []
;T5:     unreachable
;T5:   bogus:
;T5:     cleanuppad within %cs []
;T5:     ; CHECK5: CleanupPadInst has an invalid parent
;T5:     unreachable
;T5: }

;T6: define void @f() personality void ()* @g {
;T6:   entry:
;T6:     ret void
;T6:   switch1:
;T6:     %cs1 = catchswitch within none [label %catch1] unwind label %catch2
;T6:     ; CHECK6: Block containg CatchPadInst must be jumped to only by its catchswitch
;T6:   catch1:
;T6:     catchpad within %cs1 []
;T6:     unreachable
;T6:   switch2:
;T6:     %cs2 = catchswitch within none [label %catch2] unwind to caller
;T6:   catch2:
;T6:     catchpad within %cs2 []
;T6:     unreachable
;T6: }

;T7: define void @f() personality void ()* @g {
;T7:   entry:
;T7:     ret void
;T7:   switch1:
;T7:     %cs1 = catchswitch within none [label %catch1] unwind to caller
;T7:   catch1:
;T7:     catchpad within %cs1 []
;T7:     unreachable
;T7:   switch2:
;T7:     %cs2 = catchswitch within %cs1 [label %catch2] unwind to caller
;T7:     ; CHECK7: CatchSwitchInst has an invalid parent
;T7:   catch2:
;T7:     catchpad within %cs2 []
;T7:     unreachable
;T7: }

;T8: define void @f() personality void ()* @g {
;T8:   entry:
;T8:     ret void
;T8:   switch1:
;T8:     %cs1 = catchswitch within none [ label %switch1 ] unwind to caller
;T8:     ; CHECK8: CatchSwitchInst handlers must be catchpads
;T8: }
