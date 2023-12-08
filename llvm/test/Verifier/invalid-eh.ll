; UNSUPPORTED: system-windows

; RUN: sed -e s/.T1:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK1 %s
; RUN: sed -e s/.T2:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK2 %s
; RUN: sed -e s/.T3:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK3 %s
; RUN: sed -e s/.T4:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK4 %s
; RUN: sed -e s/.T5:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK5 %s
; RUN: sed -e s/.T6:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK6 %s
; RUN: sed -e s/.T7:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK7 %s
; RUN: sed -e s/.T8:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK8 %s
; RUN: sed -e s/.T9:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK9 %s
; RUN: sed -e s/.T10:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK10 %s
; RUN: sed -e s/.T11:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK11 %s
; RUN: sed -e s/.T12:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK12 %s
; RUN: sed -e s/.T13:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK13 %s
; RUN: sed -e s/.T14:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK14 %s
; RUN: sed -e s/.T15:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK15 %s
; RUN: sed -e s/.T16:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK16 %s
; RUN: sed -e s/.T17:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK17 %s
; RUN: sed -e s/.T18:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK18 %s
; RUN: sed -e s/.T19:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK19 %s
; RUN: sed -e s/.T20:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK20 %s
; RUN: sed -e s/.T21:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK21 %s
; RUN: sed -e s/.T22:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK22 %s
; RUN: sed -e s/.T23:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK23 %s
; RUN: sed -e s/.T24:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK24 %s
; RUN: sed -e s/.T25:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK25 %s
; RUN: sed -e s/.T26:// %s | not opt -passes=verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK26 %s

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

;T5: define void @f() personality ptr @g {
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

;T6: define void @f() personality ptr @g {
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

;T7: define void @f() personality ptr @g {
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

;T8: define void @f() personality ptr @g {
;T8:   entry:
;T8:     ret void
;T8:   switch1:
;T8:     %cs1 = catchswitch within none [ label %switch1 ] unwind to caller
;T8:     ; CHECK8: CatchSwitchInst handlers must be catchpads
;T8: }

;T9: define void @f() personality ptr @g {
;T9:   entry:
;T9:     ret void
;T9:   cleanup:
;T9:     %cp = cleanuppad within none []
;T9:     invoke void @g() [ "funclet"(token %cp) ]
;T9:       to label %exit unwind label %cleanup
;T9:       ; CHECK9: EH pad cannot handle exceptions raised within it
;T9:       ; CHECK9-NEXT: %cp = cleanuppad within none []
;T9:       ; CHECK9-NEXT: invoke void @g() [ "funclet"(token %cp) ]
;T9:   exit:
;T9:     ret void
;T9: }

;T10: define void @f() personality ptr @g {
;T10:   entry:
;T10:     ret void
;T10:   cleanup1:
;T10:     %cp1 = cleanuppad within none []
;T10:     unreachable
;T10:   switch:
;T10:     %cs = catchswitch within %cp1 [label %catch] unwind to caller
;T10:   catch:
;T10:     %catchp1 = catchpad within %cs [i32 1]
;T10:     unreachable
;T10:   cleanup2:
;T10:     %cp2 = cleanuppad within %catchp1 []
;T10:     unreachable
;T10:   cleanup3:
;T10:     %cp3 = cleanuppad within %cp2 []
;T10:     cleanupret from %cp3 unwind label %switch
;T10:       ; CHECK10: EH pad cannot handle exceptions raised within it
;T10:       ; CHECK10-NEXT: %cs = catchswitch within %cp1 [label %catch] unwind to caller
;T10:       ; CHECK10-NEXT: cleanupret from %cp3 unwind label %switch
;T10: }

;T11: define void @f() personality ptr @g {
;T11:   entry:
;T11:     ret void
;T11:   cleanup1:
;T11:     %cp1 = cleanuppad within none []
;T11:     unreachable
;T11:   cleanup2:
;T11:     %cp2 = cleanuppad within %cp1 []
;T11:     unreachable
;T11:   switch:
;T11:     %cs = catchswitch within none [label %catch] unwind label %cleanup2
;T11:     ; CHECK11: A single unwind edge may only enter one EH pad
;T11:     ; CHECK11-NEXT: %cs = catchswitch within none [label %catch] unwind label %cleanup2
;T11:   catch:
;T11:     catchpad within %cs [i32 1]
;T11:     unreachable
;T11: }

;T12: define void @f() personality ptr @g {
;T12:   entry:
;T12:     ret void
;T12:   cleanup:
;T12:     %cp = cleanuppad within none []
;T12:     cleanupret from %cp unwind label %switch
;T12:     ; CHECK12: A cleanupret must exit its cleanup
;T12:     ; CHECK12-NEXT: cleanupret from %cp unwind label %switch
;T12:   switch:
;T12:     %cs = catchswitch within %cp [label %catch] unwind to caller
;T12:   catch:
;T12:     catchpad within %cs [i32 1]
;T12:     unreachable
;T12: }

;T13: define void @f() personality ptr @g {
;T13:   entry:
;T13:     ret void
;T13:   switch:
;T13:     %cs = catchswitch within none [label %catch] unwind label %switch
;T13:     ; CHECK13: EH pad cannot handle exceptions raised within it
;T13:     ; CHECK13-NEXT:  %cs = catchswitch within none [label %catch] unwind label %switch
;T13:   catch:
;T13:     catchpad within %cs [i32 0]
;T13:     unreachable
;T13: }

;T14: define void @f() personality ptr @g {
;T14:   entry:
;T14:     ret void
;T14:   cleanup:
;T14:     %cp = cleanuppad within none []
;T14:     unreachable
;T14:   left:
;T14:     cleanupret from %cp unwind label %switch
;T14:   right:
;T14:     cleanupret from %cp unwind to caller
;T14:     ; CHECK14: Unwind edges out of a funclet pad must have the same unwind dest
;T14:     ; CHECK14-NEXT: %cp = cleanuppad within none []
;T14:     ; CHECK14-NEXT: cleanupret from %cp unwind label %switch
;T14:     ; CHECK14-NEXT: cleanupret from %cp unwind to caller
;T14:   switch:
;T14:     %cs = catchswitch within none [label %catch] unwind to caller
;T14:   catch:
;T14:     catchpad within %cs [i32 1]
;T14:     unreachable
;T14: }

;T15: define void @f() personality ptr @g {
;T15:   entry:
;T15:     ret void
;T15:   switch:
;T15:     %cs = catchswitch within none [label %catch] unwind to caller
;T15:   catch:
;T15:     %catch.pad = catchpad within %cs [i32 1]
;T15:     invoke void @g() [ "funclet"(token %catch.pad) ]
;T15:       to label %unreachable unwind label %target1
;T15:   unreachable:
;T15:     unreachable
;T15:   target1:
;T15:     cleanuppad within none []
;T15:     unreachable
;T15:   target2:
;T15:     cleanuppad within none []
;T15:     unreachable
;T15:   nested.1:
;T15:     %nested.pad.1 = cleanuppad within %catch.pad []
;T15:     unreachable
;T15:   nested.2:
;T15:     %nested.pad.2 = cleanuppad within %nested.pad.1 []
;T15:     cleanupret from %nested.pad.2 unwind label %target2
;T15:     ; CHECK15: Unwind edges out of a funclet pad must have the same unwind dest
;T15:     ; CHECK15-NEXT: %catch.pad = catchpad within %cs [i32 1]
;T15:     ; CHECK15-NEXT: cleanupret from %nested.pad.2 unwind label %target2
;T15:     ; CHECK15-NEXT: invoke void @g() [ "funclet"(token %catch.pad) ]
;T15:     ; CHECK15-NEXT:   to label %unreachable unwind label %target1
;T15: }

;T16: define void @f() personality ptr @g {
;T16:   entry:
;T16:     ret void
;T16:   switch:
;T16:     %cs = catchswitch within none [label %catch] unwind to caller
;T16:   catch:
;T16:     %catch.pad = catchpad within %cs [i32 1]
;T16:     invoke void @g() [ "funclet"(token %catch.pad) ]
;T16:       to label %unreachable unwind label %target1
;T16:     ; CHECK16: Unwind edges out of a catch must have the same unwind dest as the parent catchswitch
;T16:     ; CHECK16-NEXT:   %catch.pad = catchpad within %cs [i32 1]
;T16:     ; CHECK16-NEXT:  invoke void @g() [ "funclet"(token %catch.pad) ]
;T16:     ; CHECK16-NEXT:          to label %unreachable unwind label %target1
;T16:     ; CHECK16-NEXT:  %cs = catchswitch within none [label %catch] unwind to caller
;T16:   unreachable:
;T16:     unreachable
;T16:   target1:
;T16:     cleanuppad within none []
;T16:     unreachable
;T16: }

;T17: define void @f() personality ptr @g {
;T17:   entry:
;T17:     ret void
;T17:   switch:
;T17:     %cs = catchswitch within none [label %catch] unwind label %target1
;T17:   catch:
;T17:     %catch.pad = catchpad within %cs [i32 1]
;T17:     invoke void @g() [ "funclet"(token %catch.pad) ]
;T17:       to label %unreachable unwind label %target2
;T17:     ; CHECK17: Unwind edges out of a catch must have the same unwind dest as the parent catchswitch
;T17:     ; CHECK17-NEXT:  %catch.pad = catchpad within %cs [i32 1]
;T17:     ; CHECK17-NEXT:  invoke void @g() [ "funclet"(token %catch.pad) ]
;T17:     ; CHECK17-NEXT:          to label %unreachable unwind label %target2
;T17:     ; CHECK17-NEXT:  %cs = catchswitch within none [label %catch] unwind label %target1
;T17:   unreachable:
;T17:     unreachable
;T17:   target1:
;T17:     cleanuppad within none []
;T17:     unreachable
;T17:   target2:
;T17:     cleanuppad within none []
;T17:     unreachable
;T17: }

;T18: define void @f() personality ptr @g {
;T18:   entry:
;T18:     invoke void @g()
;T18:       to label %invoke.cont unwind label %left
;T18:   invoke.cont:
;T18:     invoke void @g()
;T18:       to label %unreachable unwind label %right
;T18:   left:
;T18:     %cp.left = cleanuppad within none []
;T18:     invoke void @g() [ "funclet"(token %cp.left) ]
;T18:       to label %unreachable unwind label %right
;T18:   right:
;T18:     %cp.right = cleanuppad within none []
;T18:     invoke void @g() [ "funclet"(token %cp.right) ]
;T18:       to label %unreachable unwind label %left
;T18:     ; CHECK18: EH pads can't handle each other's exceptions
;T18:     ; CHECK18-NEXT: %cp.left = cleanuppad within none []
;T18:     ; CHECK18-NEXT:  invoke void @g() [ "funclet"(token %cp.left) ]
;T18:     ; CHECK18-NEXT:          to label %unreachable unwind label %right
;T18:     ; CHECK18-NEXT:  %cp.right = cleanuppad within none []
;T18:     ; CHECK18-NEXT:  invoke void @g() [ "funclet"(token %cp.right) ]
;T18:     ; CHECK18-NEXT:          to label %unreachable unwind label %left
;T18:   unreachable:
;T18:     unreachable
;T18: }

;T19: define void @f() personality ptr @g {
;T19:   entry:
;T19:     ret void
;T19:   red:
;T19:     %redpad = cleanuppad within none []
;T19:     unreachable
;T19:   red.inner:
;T19:     %innerpad = cleanuppad within %redpad []
;T19:     invoke void @g() [ "funclet"(token %innerpad) ]
;T19:       to label %unreachable unwind label %green
;T19:   green:
;T19:     %greenswitch = catchswitch within none [label %catch] unwind label %blue
;T19:   catch:
;T19:     catchpad within %greenswitch [i32 42]
;T19:     unreachable
;T19:   blue:
;T19:     %bluepad = cleanuppad within none []
;T19:     cleanupret from %bluepad unwind label %red
;T19:     ; CHECK19: EH pads can't handle each other's exceptions
;T19:     ; CHECK19-NEXT: %redpad = cleanuppad within none []
;T19:     ; CHECK19-NEXT: invoke void @g() [ "funclet"(token %innerpad) ]
;T19:     ; CHECK19-NEXT:         to label %unreachable unwind label %green
;T19:     ; CHECK19-NEXT: %greenswitch = catchswitch within none [label %catch] unwind label %blue
;T19:     ; CHECK19-NEXT: %bluepad = cleanuppad within none []
;T19:     ; CHECK19-NEXT: cleanupret from %bluepad unwind label %red
;T19:   unreachable:
;T19:     unreachable
;T19: }

;T20: define void @f() personality ptr @g {
;T20:   entry:
;T20:     ret void
;T20:   switch:
;T20:     %cs = catchswitch within none [label %catch] unwind label %catch
;T20:     ; CHECK20: Catchswitch cannot unwind to one of its catchpads
;T20:     ; CHECK20-NEXT: %cs = catchswitch within none [label %catch] unwind label %catch
;T20:     ; CHECK20-NEXT: %cp = catchpad within %cs [i32 4]
;T20:   catch:
;T20:     %cp = catchpad within %cs [i32 4]
;T20:     unreachable
;T20: }

;T21: define void @f() personality ptr @g {
;T21:   entry:
;T21:     ret void
;T21:   switch:
;T21:     %cs = catchswitch within none [label %catch1] unwind label %catch2
;T21:     ; CHECK21: Catchswitch cannot unwind to one of its catchpads
;T21:     ; CHECK21-NEXT: %cs = catchswitch within none [label %catch1] unwind label %catch2
;T21:     ; CHECK21-NEXT: %cp2 = catchpad within %cs [i32 2]
;T21:   catch1:
;T21:     %cp1 = catchpad within %cs [i32 1]
;T21:     unreachable
;T21:   catch2:
;T21:     %cp2 = catchpad within %cs [i32 2]
;T21:     unreachable
;T21: }

;T22: define void @f() personality ptr @g {
;T22:   invoke void @g()
;T22:           to label %merge unwind label %cleanup
;T22:
;T22: cleanup:
;T22:   %outer = cleanuppad within none []
;T22:   invoke void @g() [ "funclet"(token %outer) ]
;T22:           to label %merge unwind label %merge
;T22:   ; CHECK22: The unwind destination does not have an exception handling instruction!
;T22:   ; CHECK22:   invoke void @g() [ "funclet"(token %outer) ]
;T22:   ; CHECK22:           to label %merge unwind label %merge
;T22:
;T22: merge:
;T22:   unreachable
;T22: }

;T23: define void @f() personality ptr @g {
;T23:   invoke void @g()
;T23:           to label %exit unwind label %pad
;T23:
;T23: pad:
;T23:   %outer = catchpad within %outer []
;T23:   ; CHECK23: CatchPadInst needs to be directly nested in a CatchSwitchInst.
;T23:   ; CHECK23:   %outer = catchpad within %outer []
;T23:   unreachable
;T23:
;T23: exit:
;T23:   unreachable
;T23: }

;T24: define void @f() personality ptr @g {
;T24:   invoke void @g()
;T24:           to label %exit unwind label %pad
;T24:   ; CHECK24: A single unwind edge may only enter one EH pad
;T24:   ; CHECK24:   invoke void @g()
;T24:   ; CHECK24:           to label %exit unwind label %pad
;T24:
;T24: pad:
;T24:   %outer = cleanuppad within %outer []
;T24:   ; CHECK24: FuncletPadInst must not be nested within itself
;T24:   ; CHECK24:   %outer = cleanuppad within %outer []
;T24:   unreachable
;T24:
;T24: exit:
;T24:   unreachable
;T24: }

;T25: define void @f() personality ptr @g {
;T25: entry:
;T25:   unreachable
;T25:
;T25: catch.dispatch:
;T25:   %cs = catchswitch within %cp2 [label %catch] unwind label %ehcleanup
;T25:   ; CHECK25: EH pad jumps through a cycle of pads
;T25:   ; CHECK25:   %cs = catchswitch within %cp2 [label %catch] unwind label %ehcleanup
;T25:
;T25: catch:
;T25:   %cp2 = catchpad within %cs [ptr null, i32 64, ptr null]
;T25:   unreachable
;T25:
;T25: ehcleanup:
;T25:   %cp3 = cleanuppad within none []
;T25:   cleanupret from %cp3 unwind to caller
;T25: }

;T26: define void @f() personality ptr @g {
;T26: entry:
;T26:   ret void
;T26:
;T26: ehcleanup:
;T26:   cleanuppad within none []
;T26:   cleanupret from none unwind label %ehcleanup
;T26:   ; CHECK26: A cleanupret must exit its cleanup
;T26:   ; CHECK26:   cleanupret from none unwind label %ehcleanup
;T26:   ; CHECK26: CleanupReturnInst needs to be provided a CleanupPad
;T26:   ; CHECK26:   cleanupret from none unwind label %ehcleanup
;T26:   ; CHECK26: token none
;T26: }
