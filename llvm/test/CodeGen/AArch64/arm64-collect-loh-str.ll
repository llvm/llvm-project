; RUN: llc -o - %s -mtriple=arm64-apple-ios -O2 | FileCheck %s
; RUN: llc -o - %s -mtriple=arm64_32-apple-ios -O2 | FileCheck %s
; Test case for <rdar://problem/15942912>.
; AdrpAddStr cannot be used when the store uses same
; register as address and value. Indeed, the related
; if applied, may completely remove the definition or
; at least provide a wrong one (with the offset folded
; into the definition).

%struct.anon = type { ptr, ptr }

@pptp_wan_head = internal global %struct.anon zeroinitializer, align 8

; CHECK-LABEL: _pptp_wan_init
; CHECK: ret
; CHECK-NOT: AdrpAddStr
define i32 @pptp_wan_init() {
entry:
  store ptr null, ptr @pptp_wan_head, align 8
  store ptr @pptp_wan_head, ptr getelementptr inbounds (%struct.anon, ptr @pptp_wan_head, i64 0, i32 1), align 8
  ret i32 0
}


