; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -stats 2>&1 \
; RUN:   -hwasan-skip-hot-code=0 | FileCheck %s --check-prefix=FULL
; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -stats 2>&1 \
; RUN:   -hwasan-skip-hot-code=1 | FileCheck %s --check-prefix=SELSAN

; REQUIRES: asserts

; FULL: 1 hwasan - Number of HWASAN instrumented funcs
; FULL: 1 hwasan - Number of total funcs HWASAN

; SELSAN: 1 hwasan - Number of HWASAN instrumented funcs
; SELSAN: 1 hwasan - Number of HWASAN funcs without PS
; SELSAN: 1 hwasan - Number of total funcs HWASAN

define void @not_sanitized() { ret void }
define void @sanitized_no_ps() sanitize_hwaddress { ret void }
