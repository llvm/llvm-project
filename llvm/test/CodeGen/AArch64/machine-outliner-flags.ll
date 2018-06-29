; RUN: llc %s -debug-pass=Structure -verify-machineinstrs \
; RUN: -enable-machine-outliner -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=ENABLE

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs \
; RUN: -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=NOT-ADDED

; RUN: llc %s -O=0 -debug-pass=Structure -verify-machineinstrs \
; RUN: -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=OPTNONE

; Make sure that the outliner is added to the pass pipeline only when the
; appropriate flags/settings are set. Make sure it isn't added otherwise.
;
; Cases where it should be added:
;  * -enable-machine-outliner
;
; Cases where it should not be added:
;  * -O0 or equivalent
;  * -enable-machine-outliner is not passed

; ENABLE: Machine Outliner
; NOT-ADDED-NOT: Machine Outliner
; OPTNONE-NOT: Machine Outliner

define void @foo() {
  ret void;
}

