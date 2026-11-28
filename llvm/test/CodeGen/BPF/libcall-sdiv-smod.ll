; RUN: not llc < %s -mtriple=bpf 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc < %s -mtriple=bpf -mcpu=v4 -disable-sdiv-smod 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: not llc < %s -mtriple=bpf -mcpu=v5 -disable-sdiv-smod 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: llc < %s -mtriple=bpf -bpf-allows-libcalls | FileCheck %s --check-prefix=ALU32
; RUN: llc < %s -mtriple=bpf -mcpu=v5 -disable-sdiv-smod -bpf-allows-libcalls | FileCheck %s --check-prefix=ALU32
; RUN: llc < %s -mtriple=bpf -mcpu=v4 -disable-sdiv-smod -bpf-allows-libcalls | FileCheck %s --check-prefix=ALU32
; RUN: llc < %s -mtriple=bpf -mcpu=v3 -bpf-allows-libcalls | FileCheck %s --check-prefix=ALU32
; RUN: llc < %s -mtriple=bpf -mcpu=v2 -bpf-allows-libcalls | FileCheck %s --check-prefix=NOALU32
; RUN: llc < %s -mtriple=bpf -mcpu=v1 -bpf-allows-libcalls | FileCheck %s --check-prefix=NOALU32

; sdiv/smod are rejected when the target doesn't support it natively
; (v1/v2/v3, or v4/v5 with -disable-sdiv-smod). With -bpf-allows-libcalls
; they are lowered to the __divdi3/__divsi3 and __moddi3/__modsi3 libcalls.

; ERR: unsupported signed division

define i64 @sdiv_i64(i64 %a, i64 %b) {
  %r = sdiv i64 %a, %b
  ret i64 %r
}

; ALU32-LABEL: sdiv_i64:
; ALU32:       call __divdi3
; NOALU32-LABEL: sdiv_i64:
; NOALU32:       call __divdi3

define i64 @srem_i64(i64 %a, i64 %b) {
  %r = srem i64 %a, %b
  ret i64 %r
}

; ALU32-LABEL: srem_i64:
; ALU32:       call __moddi3
; NOALU32-LABEL: srem_i64:
; NOALU32:       call __moddi3

define i32 @sdiv_i32(i32 %a, i32 %b) {
  %r = sdiv i32 %a, %b
  ret i32 %r
}

; ALU32-LABEL: sdiv_i32:
; ALU32:       call __divsi3
; NOALU32-LABEL: sdiv_i32:
; NOALU32:       call __divdi3

define i32 @srem_i32(i32 %a, i32 %b) {
  %r = srem i32 %a, %b
  ret i32 %r
}

; ALU32-LABEL: srem_i32:
; ALU32:       call __modsi3
; NOALU32-LABEL: srem_i32:
; NOALU32:       call __moddi3