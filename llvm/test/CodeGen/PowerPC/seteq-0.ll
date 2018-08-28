; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

define i32 @eq0(i32 %a) {
        %tmp.1 = icmp eq i32 %a, 0              ; <i1> [#uses=1]
        %tmp.2 = zext i1 %tmp.1 to i32          ; <i32> [#uses=1]
        ret i32 %tmp.2

; CHECK: cntlzw [[REG:[0-9]+]], 3
; CHECK: rlwinm 3, [[REG]], 27, 31, 31
; CHECK: blr
}

