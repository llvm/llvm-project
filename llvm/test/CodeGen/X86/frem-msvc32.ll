; Make sure that 32-bit FREM is promoted to 64-bit FREM on 32-bit MSVC.

; MSVC does not have a 32-bit fmodf function, so it must be promoted to
; a 64-bit fmod rtlib call.
; RUN: llc -mtriple=i686-pc-windows-msvc -O0 < %s  | FileCheck %s

; CHECK: @do_frem32
; CHECK: {{_fmod$}}
define float @do_frem32(float %a, float %b) {
    %val = frem float %a, %b
    ret float %val
}
