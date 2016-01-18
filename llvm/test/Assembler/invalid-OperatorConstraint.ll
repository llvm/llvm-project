; RUN: sed -e s/.T1:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK1 %s
; RUN: sed -e s/.T2:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK2 %s
; RUN: sed -e s/.T3:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK3 %s
; RUN: sed -e s/.T4:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK4 %s
; RUN: sed -e s/.T5:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK5 %s
; RUN: sed -e s/.T6:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK6 %s
; RUN: sed -e s/.T7:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK7 %s
; RUN: sed -e s/.T8:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK8 %s
; RUN: sed -e s/.T9:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK9 %s

;T1: define void @f() {
;T1:   entry:
;T1:     ; operator constraint requires an operator
;T1:     catchret undef to label %entry
;T1:     ; CHECK1: [[@LINE-1]]:15: error: Catchpad value required in this position
;T1: }

;T2: define void @f() {
;T2:   entry:
;T2:     %x = cleanuppad []
;T2:     ; catchret's first operand's operator must be catchpad
;T2:     catchret %x to label %entry
;T2:     ; CHECK2: [[@LINE-1]]:15: error: '%x' is not a catchpad
;T2: }

;T3: define void @f() {
;T3:   entry:
;T3:     ; catchret's first operand's operator must be catchpad
;T3:     ; (forward reference case)
;T3:     catchret %x to label %next
;T3:     ; CHECK3: [[@LINE-1]]:15: error: '%x' is not a catchpad
;T3:   next:
;T3:     %x = cleanuppad []
;T3:     ret void
;T3: }

;T4: define void @f() {
;T4:   entry:
;T4:     ; operator constraint requires an operator
;T4:     cleanupret undef unwind label %entry
;T4:     ; CHECK4: [[@LINE-1]]:17: error: Cleanuppad value required in this position
;T4: }

;T5: define void @f() {
;T5:   entry:
;T5:     %x = catchpad []
;T5:             to label %next unwind label %entry
;T5:   next:
;T5:     ; cleanupret first operand's operator must be cleanuppad
;T5:     cleanupret %x unwind to caller
;T5:     ; CHECK5: [[@LINE-1]]:17: error: '%x' is not a cleanuppad
;T5: }

;T6: define void @f() {
;T6:   entry:
;T6:     ; cleanupret's first operand's operator must be cleanuppad
;T6:     ; (forward reference case)
;T6:     cleanupret %x unwind label %next
;T6:     ; CHECK6: [[@LINE-1]]:17: error: '%x' is not a cleanuppad
;T6:   next:
;T6:     %x = catchpad [] to label %entry unwind label %next
;T6: }

;T7: define void @f() {
;T7:   entry:
;T7:     ; operator constraint requires an operator
;T7:     cleanupendpad undef unwind to caller
;T7:     ; CHECK7: [[@LINE-1]]:20: error: Cleanuppad value required in this position
;T7: }

;T8: define void @f() {
;T8:   entry:
;T8:     %x = catchpad []
;T8:             to label %next unwind label %entry
;T8:   next:
;T8:     ; cleanupret first operand's operator must be cleanuppad
;T8:     cleanupendpad %x unwind label next
;T8:     ; CHECK8: [[@LINE-1]]:20: error: '%x' is not a cleanuppad
;T8: }

;T9: define void @f() {
;T9:   entry:
;T9:     ; cleanupret's first operand's operator must be cleanuppad
;T9:     ; (forward reference case)
;T9:     cleanupendpad %x unwind label %next
;T9:     ; CHECK9: [[@LINE-1]]:20: error: '%x' is not a cleanuppad
;T9:   next:
;T9:     %x = catchpad [] to label %entry unwind label %next
;T9: }
