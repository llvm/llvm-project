; RUN: llc -mtriple=x86_64-- -enable-new-pm -debug-pass-manager -stop-after=verify,1 \
; RUN:      %s -o /dev/null 2>&1 | FileCheck -check-prefix=STOP-AFTER-1 %s

; RUN: llc -mtriple=x86_64-- -enable-new-pm -debug-pass-manager -stop-after=verify,0 \
; RUN:      %s -o /dev/null 2>&1 | FileCheck -check-prefix=STOP-AFTER-0 %s

; RUN: llc -mtriple=x86_64-- -enable-new-pm -debug-pass-manager -stop-before=verify,1 \
; RUN:      %s -o /dev/null 2>&1 | FileCheck -check-prefix=STOP-BEFORE-1 %s

; RUN: llc -mtriple=x86_64-- -enable-new-pm -debug-pass-manager -start-before=verify,1 \
; RUN:      %s -o /dev/null 2>&1 | FileCheck -check-prefix=START-BEFORE-1 %s

; RUN: llc -mtriple=x86_64-- -enable-new-pm -debug-pass-manager -start-after=verify,1 \
; RUN:      %s -o /dev/null 2>&1 | FileCheck -check-prefix=START-AFTER-1 %s


; STOP-AFTER-1: Running pass: VerifierPass
; STOP-AFTER-1: Running pass: VerifierPass

; STOP-AFTER-0-NOT: Running pass: VerifierPass
; STOP-AFTER-0: Running pass: VerifierPass
; STOP-AFTER-0-NOT: Running pass: VerifierPass

; STOP-BEFORE-1: Running pass: VerifierPass
; STOP-BEFORE-1-NOT: Running pass: VerifierPass

; START-BEFORE-1-NOT: Running pass: VerifierPass
; START-BEFORE-1: Running pass: VerifierPass
; START-BEFORE-1-NOT: Running pass: VerifierPass

; START-AFTER-1-NOT: Running pass: VerifierPass

define void @f() {
  ret void
}
