; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | not grep rlwinm

; FIXME: This optimization has temporarily regressed with crbits enabled by
; default at the default CodeOpt level.
; XFAIL: *

define i32 @setcc_one_or_zero(ptr %a) {
entry:
        %tmp.1 = icmp ne ptr %a, null          ; <i1> [#uses=1]
        %inc.1 = zext i1 %tmp.1 to i32          ; <i32> [#uses=1]
        ret i32 %inc.1
}

