; REQUIRES: amdgpu-registered-target

; AMDGPU inserts the fourth instance of dead-mi-elimination pass after detect-dead-lanes
; This checks that the pipeline works properly with -{start|stop}-{after|before} options.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -enable-new-pm -stop-before=dead-mi-elimination,4 --print-pipeline-passes -filetype=null %s | FileCheck %s --check-prefix=STOP-BEFORE-4
; STOP-BEFORE-4: dead-mi-elimination
; STOP-BEFORE-4: dead-mi-elimination
; STOP-BEFORE-4: dead-mi-elimination
; STOP-BEFORE-4-NOT: dead-mi-elimination

; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -enable-new-pm -start-after=dead-mi-elimination,4 --print-pipeline-passes -filetype=null %s | FileCheck %s --check-prefix=START-AFTER-4
; START-AFTER-4: require<reg-usage>
; START-AFTER-4: init-undef
; START-AFTER-4: reg-usage-collector
; START-AFTER-4-NOT: dead-mi-elimination

; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -enable-new-pm -start-after=dead-mi-elimination,3 --print-pipeline-passes -filetype=null %s | FileCheck %s --check-prefix=START-AFTER-3
; START-AFTER-3: require<reg-usage>
; START-AFTER-3: detect-dead-lanes
; START-AFTER-3: dead-mi-elimination
; START-AFTER-3-NOT: dead-mi-elimination

; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -enable-new-pm -stop-after=dead-mi-elimination,4 --print-pipeline-passes -filetype=null %s | FileCheck %s --check-prefix=STOP-AFTER-4
; STOP-AFTER-4: dead-mi-elimination
; STOP-AFTER-4: dead-mi-elimination
; STOP-AFTER-4: dead-mi-elimination
; STOP-AFTER-4: detect-dead-lanes
; STOP-AFTER-4: dead-mi-elimination
; STOP-AFTER-4-NOT: init-undef
; STOP-AFTER-4: free-machine-function

; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -enable-new-pm -stop-after=dead-mi-elimination,3 --print-pipeline-passes -filetype=null %s | FileCheck %s --check-prefix=STOP-AFTER-3
; STOP-AFTER-3: dead-mi-elimination
; STOP-AFTER-3: dead-mi-elimination
; STOP-AFTER-3: dead-mi-elimination
; STOP-AFTER-3-NOT: si-shrink-instructions
; STOP-AFTER-3: require<reg-usage>
; STOP-AFTER-3-NOT: dead-mi-elimination

; RUN: llc -mtriple=amdgcn-amd-amdhsa -O3 -enable-new-pm -start-after=dead-mi-elimination,3 -stop-after=dead-mi-elimination,4 --print-pipeline-passes -filetype=null %s | FileCheck %s --check-prefix=START-AFTER-3-STOP-AFTER-4
; START-AFTER-3-STOP-AFTER-4: detect-dead-lanes
; START-AFTER-3-STOP-AFTER-4: dead-mi-elimination
; START-AFTER-3-STOP-AFTER-4-NOT: init-undef
; START-AFTER-3-STOP-AFTER-4: free-machine-function
