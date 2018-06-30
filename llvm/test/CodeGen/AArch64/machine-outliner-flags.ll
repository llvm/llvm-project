; REQUIRES: asserts
; RUN: llc %s -debug-pass=Structure -verify-machineinstrs \
; RUN: --debug-only=machine-outliner -enable-machine-outliner=always \
; RUN: -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=ALWAYS

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs \
; RUN: --debug-only=machine-outliner -enable-machine-outliner \
; RUN: -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=ENABLE

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs \
; RUN: -enable-machine-outliner=never -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=NEVER

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
;  * -enable-machine-outliner=always
;
; Cases where it should not be added:
;  * -O0 or equivalent
;  * -enable-machine-outliner is not passed
;  * -enable-machine-outliner=never is passed

; ALWAYS: Machine Outliner
; ALWAYS: Machine Outliner: Running on all functions
; ENABLE: Machine Outliner
; ENABLE: Machine Outliner: Running on all functions
; NEVER-NOT: Machine Outliner
; NOT-ADDED-NOT: Machine Outliner
; OPTNONE-NOT: Machine Outliner

define void @foo() {
  ret void;
}

