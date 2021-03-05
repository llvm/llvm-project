; This file tests 64 bit TLS variable generation

; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple \
; RUN:      powerpc64-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple \
; RUN:      powerpc64-ibm-aix-xcoff -data-sections=false < %s | FileCheck %s \
; RUN:      --check-prefix=NODATASEC

; When data-sections is true (default), we emit data into separate sections.
; When data-sections is false, we emit data into the .data / .tdata sections.

; Long long global variable, TLS/Non-TLS, local/weak linkage

; CHECK:           .csect  global_long_long_internal_val_initialized[RW],3
; CHECK-NEXT:      .lglobl global_long_long_internal_val_initialized[RW]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  8, 1
; NODATASEC:       .csect  .data[RW],3
; NODATASEC-NEXT:  .lglobl global_long_long_internal_val_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:global_long_long_internal_val_initialized:
; NODATASEC-NEXT:  .vbyte  8, 1
@global_long_long_internal_val_initialized = internal global i64 1, align 8

; CHECK-NEXT:      .csect  tls_global_long_long_internal_val_initialized[TL],3
; CHECK-NEXT:      .lglobl tls_global_long_long_internal_val_initialized[TL]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  8, 1
; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .lglobl tls_global_long_long_internal_val_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:tls_global_long_long_internal_val_initialized:
; NODATASEC-NEXT:  .vbyte  8, 1
@tls_global_long_long_internal_val_initialized = internal thread_local global i64 1, align 8

; CHECK-NEXT:      .lcomm  global_long_long_internal_zero_initialized,8,global_long_long_internal_zero_initialized[BS],3
; NODATASEC-NEXT:  .lcomm  global_long_long_internal_zero_initialized,8,global_long_long_internal_zero_initialized[BS],3
@global_long_long_internal_zero_initialized = internal global i64 0, align 8

; CHECK-NEXT:      .lcomm  tls_global_long_long_internal_zero_initialized,8,tls_global_long_long_internal_zero_initialized[UL],3
; NODATASEC-NEXT:  .lcomm  tls_global_long_long_internal_zero_initialized,8,tls_global_long_long_internal_zero_initialized[UL],3
@tls_global_long_long_internal_zero_initialized = internal thread_local global i64 0, align 8

; CHECK-NEXT:      .csect  global_long_long_weak_val_initialized[RW],3
; CHECK-NEXT:      .weak   global_long_long_weak_val_initialized[RW]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  8, 1
; NODATASEC-NEXT:  .csect  .data[RW],3
; NODATASEC-NEXT:  .weak   global_long_long_weak_val_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:global_long_long_weak_val_initialized:
; NODATASEC-NEXT:  .vbyte  8, 1
@global_long_long_weak_val_initialized = weak global i64 1, align 8

; CHECK-NEXT:      .csect  tls_global_long_long_weak_val_initialized[TL],3
; CHECK-NEXT:      .weak   tls_global_long_long_weak_val_initialized[TL]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  8, 1
; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .weak   tls_global_long_long_weak_val_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:tls_global_long_long_weak_val_initialized:
; NODATASEC-NEXT:  .vbyte  8, 1
@tls_global_long_long_weak_val_initialized = weak thread_local global i64 1, align 8

; CHECK-NEXT:      .csect  global_long_long_weak_zero_initialized[RW],3
; CHECK-NEXT:      .weak   global_long_long_weak_zero_initialized[RW]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  8, 0
; NODATASEC-NEXT:  .csect  .data[RW],3
; NODATASEC-NEXT:  .weak   global_long_long_weak_zero_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:global_long_long_weak_zero_initialized:
; NODATASEC-NEXT:  .vbyte  8, 0
@global_long_long_weak_zero_initialized = weak global i64 0, align 8

; CHECK-NEXT:      .csect  tls_global_long_long_weak_zero_initialized[TL],3
; CHECK-NEXT:      .weak   tls_global_long_long_weak_zero_initialized[TL]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  8, 0
; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .weak   tls_global_long_long_weak_zero_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:tls_global_long_long_weak_zero_initialized:
; NODATASEC-NEXT:  .vbyte  8, 0
@tls_global_long_long_weak_zero_initialized = weak thread_local global i64 0, align 8
