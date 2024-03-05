; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S  \
; RUN:   -hwasan-skip-hot-code=0 | FileCheck %s --check-prefix=FULL
; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S  \
; RUN:   -hwasan-skip-hot-code=1 | FileCheck %s --check-prefix=SELSAN

; FULL: @not_sanitized
; FULL-NEXT: %x = alloca i8, i64 4
; FULL: @sanitized_no_ps
; FULL-SAME: @__hwasan_personality_thunk

; SELSAN: @not_sanitized
; SELSAN-NEXT: %x = alloca i8, i64 4
; SELSAN: @sanitized_no_ps
; SELSAN-SAME: @__hwasan_personality_thunk

declare void @use(ptr)

define void @not_sanitized() {
  %x = alloca i8, i64 4
  call void @use(ptr %x)
  ret void
 }

define void @sanitized_no_ps() sanitize_hwaddress {
  %x = alloca i8, i64 4
  call void @use(ptr %x)
  ret void
 }
