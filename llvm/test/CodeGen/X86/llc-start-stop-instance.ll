; RUN: llc -mtriple=x86_64-- -debug-pass=Structure -stop-after=dead-mi-elimination,1 %s -o /dev/null 2>&1 \
; RUN:   | FileCheck -check-prefix=STOP-AFTER-DEAD1 %s

; RUN: llc -mtriple=x86_64-- -debug-pass=Structure -stop-after=dead-mi-elimination,0 %s -o /dev/null 2>&1 \
; RUN:   | FileCheck -check-prefix=STOP-AFTER-DEAD0 %s


; RUN: llc -mtriple=x86_64-- -debug-pass=Structure -stop-before=dead-mi-elimination,1 %s -o /dev/null 2>&1 \
; RUN:   | FileCheck -check-prefix=STOP-BEFORE-DEAD1 %s


; RUN: llc -mtriple=x86_64-- -debug-pass=Structure -start-before=dead-mi-elimination,1 %s -o /dev/null 2>&1 \
; RUN:   | FileCheck -check-prefix=START-BEFORE-DEAD1 %s

; RUN: llc -mtriple=x86_64-- -debug-pass=Structure -start-after=dead-mi-elimination,1 %s -o /dev/null 2>&1 \
; RUN:   | FileCheck -check-prefix=START-AFTER-DEAD1 %s



; STOP-AFTER-DEAD1: -dead-mi-elimination
; STOP-AFTER-DEAD1-SAME: -dead-mi-elimination

; STOP-AFTER-DEAD1: Remove dead machine instructions
; STOP-AFTER-DEAD1: Remove dead machine instructions



; STOP-AFTER-DEAD0:     -dead-mi-elimination

; STOP-AFTER-DEAD0-NOT: Remove dead machine instructions
; STOP-AFTER-DEAD0: Remove dead machine instructions
; STOP-AFTER-DEAD0-NOT: Remove dead machine instructions



; STOP-BEFORE-DEAD1:     -dead-mi-elimination
; STOP-BEFORE-DEAD1: Remove dead machine instructions
; STOP-BEFORE-DEAD1-NOT: Remove dead machine instructions



; START-BEFORE-DEAD1:     -dead-mi-elimination
; START-BEFORE-DEAD1-NOT: Remove dead machine instructions
; START-BEFORE-DEAD1: Remove dead machine instructions
; START-BEFORE-DEAD1-NOT: Remove dead machine instructions



; START-AFTER-DEAD1-NOT: -dead-mi-elimination
; START-AFTER-DEAD1-NOT: Remove dead machine instructions
