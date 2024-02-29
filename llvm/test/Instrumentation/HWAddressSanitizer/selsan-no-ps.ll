; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -stats 2>&1 \
; RUN:   -hwasan-skip-hot-code=0 | FileCheck %s --check-prefix=FULL
; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -stats 2>&1 \
; RUN:   -hwasan-skip-hot-code=1 | FileCheck %s --check-prefix=SELSAN

; FULL: 1 hwasan - Number of funcs considered for HWASAN
; FULL: 1 hwasan - Number of HWASAN ctors
; FULL: 1 hwasan - Number of HWASAN instrumented funcs

; SELSAN: 1 hwasan - Number of funcs considered for HWASAN
; SELSAN: 1 hwasan - Number of HWASAN ctors
; SELSAN: 1 hwasan - Number of HWASAN instrumented funcs
; SELSAN: 1 hwasan - Number of HWASAN funcs without PS

define void @not_sanitized() { ret void }
define void @sanitized_no_ps() sanitize_hwaddress { ret void }
