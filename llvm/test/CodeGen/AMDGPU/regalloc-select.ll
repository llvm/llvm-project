; RUN: llc -mtriple=amdgcn-- -O0 -enable-new-pm -print-pipeline-passes -filetype=null %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=amdgcn-- -O0 -enable-new-pm -regalloc-npm='regallocfast<filter=sgpr;no-clear-vregs>' -print-pipeline-passes -filetype=null %s | FileCheck %s --check-prefix=CUSTOM

; DEFAULT: regallocfast<filter=sgpr>
; DEFAULT: regallocfast<filter=vgpr>

; Just a proof of concept that we can modify parameters of register allocator.
; CUSTOM: regallocfast<filter=sgpr;no-clear-vregs>
; CUSTOM: regallocfast<filter=vgpr>
