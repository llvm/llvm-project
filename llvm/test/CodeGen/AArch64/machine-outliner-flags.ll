; REQUIRES: asserts
; RUN: llc %s -debug-pass=Structure -verify-machineinstrs --debug-only=machine-outliner -enable-machine-outliner=always -mtriple arm64---- -o /dev/null 2>&1 | FileCheck %s -check-prefixes=CHECK,ALWAYS
; RUN: llc %s -debug-pass=Structure -verify-machineinstrs --debug-only=machine-outliner -enable-machine-outliner -mtriple arm64---- -o /dev/null 2>&1 | FileCheck %s -check-prefixes=CHECK,ALWAYS

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs --debug-only=machine-outliner -mtriple arm64---- -o /dev/null 2>&1 | FileCheck %s -check-prefixes=CHECK,TARGET-DEFAULT

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs --debug-only=machine-outliner -enable-machine-outliner=optimistic-pgo -mtriple arm64---- -o /dev/null 2>&1 | FileCheck %s -check-prefixes=CHECK,OPTIMISTIC

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs --debug-only=machine-outliner -enable-machine-outliner=conservative-pgo -mtriple arm64---- -o /dev/null 2>&1 | FileCheck %s -check-prefixes=CHECK,CONSERVATIVE

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs --debug-only=machine-outliner -enable-machine-outliner=never -mtriple arm64---- -o /dev/null 2>&1 | FileCheck %s -check-prefix=DISABLED
; RUN: llc %s -debug-pass=Structure -verify-machineinstrs --debug-only=machine-outliner -O=0 -mtriple arm64---- -o /dev/null 2>&1 | FileCheck %s -check-prefix=DISABLED

; Make sure that the outliner is added to the pass pipeline only when the
; appropriate flags/settings are set. Make sure it isn't added otherwise.
;
; Cases where it should be added:
;  * -enable-machine-outliner
;  * -enable-machine-outliner=always
;  * -enable-machine-outliner=optimistic-pgo
;  * -enable-machine-outliner=conservative-pgo
;  * -enable-machine-outliner is not passed (AArch64 supports target-default outlining)
;
; Cases where it should not be added:
;  * -O0 or equivalent
;  * -enable-machine-outliner=never is passed

; CHECK: Machine Outliner
; DISABLED-NOT: Machine Outliner
; ALWAYS: Machine Outliner: Running on all functions
; OPTIMISTIC: Machine Outliner: Running on optimistically cold functions
; CONSERVATIVE: Machine Outliner: Running on conservatively cold functions
; TARGET-DEFAULT: Machine Outliner: Running on target-default functions

define void @foo() {
  ret void;
}
