; RUN: llc < %s | FileCheck %s
; RUN: %if ptxas %{ llc < %s | %ptxas-verify %}

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Check that we can handle global variables of large integer type.

; (lsb) 0x0102'0304'0506...0F10 (msb)
@gv = addrspace(1) externally_initialized global i128 21345817372864405881847059188222722561, align 16
; CHECK: .visible .global .align 16 .b8 gv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
