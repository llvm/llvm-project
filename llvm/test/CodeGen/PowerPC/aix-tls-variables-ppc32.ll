; This file tests 32 bit TLS variable generation.

; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple \
; RUN:      powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple \
; RUN:      powerpc-ibm-aix-xcoff -data-sections=false < %s | FileCheck %s \
; RUN:      --check-prefix=NODATASEC

; When data-sections is true (default), we emit data into separate sections.
; When data-sections is false, we emit data into the .data / .tdata sections.

; Int global variable, TLS/Non-TLS, local/external/weak/common linkage
; CHECK:           .csect  global_int_external_val_initialized[RW],2
; CHECK-NEXT:      .globl  global_int_external_val_initialized[RW]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 1

; NODATASEC:       .csect  .data[RW],3
; NODATASEC-NEXT:  .globl  global_int_external_val_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:global_int_external_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 1
@global_int_external_val_initialized = global i32 1, align 4

; CHECK-NEXT:      .csect  global_int_external_zero_initialized[RW],2
; CHECK-NEXT:      .globl  global_int_external_zero_initialized[RW]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 0

; NODATASEC-NEXT:  .globl  global_int_external_zero_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:global_int_external_zero_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
@global_int_external_zero_initialized = global i32 0, align 4

; CHECK-NEXT:      .csect  tls_global_int_external_val_initialized[TL],2
; CHECK-NEXT:      .globl  tls_global_int_external_val_initialized[TL]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 1

; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .globl  tls_global_int_external_val_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:tls_global_int_external_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 1
@tls_global_int_external_val_initialized = thread_local global i32 1, align 4

; CHECK-NEXT:      .csect  tls_global_int_external_zero_initialized[TL],2
; CHECK-NEXT:      .globl  tls_global_int_external_zero_initialized[TL]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 0

; NODATASEC-NEXT:  .globl  tls_global_int_external_zero_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:tls_global_int_external_zero_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
@tls_global_int_external_zero_initialized = thread_local global i32 0, align 4

; CHECK-NEXT:      .csect  global_int_local_val_initialized[RW],2
; CHECK-NEXT:      .lglobl global_int_local_val_initialized[RW]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 2

; NODATASEC-NEXT:  .csect .data[RW],3
; NODATASEC-NEXT:  .lglobl global_int_local_val_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:global_int_local_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 2
@global_int_local_val_initialized = internal global i32 2, align 4

; CHECK-NEXT:      .csect  tls_global_int_local_val_initialized[TL],2
; CHECK-NEXT:      .lglobl tls_global_int_local_val_initialized[TL]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 2

; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .lglobl tls_global_int_local_val_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:tls_global_int_local_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 2
@tls_global_int_local_val_initialized = internal thread_local global i32 2, align 4

; CHECK-NEXT:      .lcomm  global_int_local_zero_initialized,4,global_int_local_zero_initialized[BS],2
; NODATASEC-NEXT:  .lcomm  global_int_local_zero_initialized,4,global_int_local_zero_initialized[BS],2
@global_int_local_zero_initialized = internal global i32 0, align 4

; CHECK-NEXT:      .lcomm  tls_global_int_local_zero_initialized,4,tls_global_int_local_zero_initialized[UL],2
; NODATASEC-NEXT:  .lcomm  tls_global_int_local_zero_initialized,4,tls_global_int_local_zero_initialized[UL],2
@tls_global_int_local_zero_initialized = internal thread_local global i32 0, align 4

; CHECK-NEXT:      .csect  global_int_weak_zero_initialized[RW],2
; CHECK-NEXT:      .weak   global_int_weak_zero_initialized[RW]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 0

; NODATASEC-NEXT:  .csect  .data[RW],3
; NODATASEC-NEXT:  .weak   global_int_weak_zero_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:global_int_weak_zero_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
@global_int_weak_zero_initialized = weak global i32 0, align 4

; CHECK-NEXT:      .csect  tls_global_int_weak_zero_initialized[TL],2
; CHECK-NEXT:      .weak   tls_global_int_weak_zero_initialized[TL]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 0

; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .weak   tls_global_int_weak_zero_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:tls_global_int_weak_zero_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
@tls_global_int_weak_zero_initialized = weak thread_local global i32 0, align 4

; CHECK-NEXT:      .comm   global_int_common_zero_initialized[RW],4,2
; NODATASEC-NEXT:  .comm   global_int_common_zero_initialized[RW],4,2
@global_int_common_zero_initialized = common global i32 0, align 4

; CHECK-NEXT:      .comm   tls_global_int_common_zero_initialized[UL],4,2
; NODATASEC-NEXT:  .comm   tls_global_int_common_zero_initialized[UL],4,2
@tls_global_int_common_zero_initialized = common thread_local global i32 0, align 4

; CHECK-NEXT:      .csect  global_int_weak_val_initialized[RW],2
; CHECK-NEXT:      .weak   global_int_weak_val_initialized[RW]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 1

; NODATASEC-NEXT:  .csect  .data[RW],3
; NODATASEC-NEXT:  .weak   global_int_weak_val_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:global_int_weak_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 1
@global_int_weak_val_initialized = weak global i32 1, align 4

; CHECK-NEXT:      .csect  tls_global_int_weak_val_initialized[TL],2
; CHECK-NEXT:      .weak   tls_global_int_weak_val_initialized[TL]
; CHECK-NEXT:      .align  2
; CHECK-NEXT:      .vbyte  4, 1

; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .weak   tls_global_int_weak_val_initialized
; NODATASEC-NEXT:  .align  2
; NODATASEC-NEXT:tls_global_int_weak_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 1
@tls_global_int_weak_val_initialized = weak thread_local global i32 1, align 4

; CHECK-NEXT:      .extern global_int_external_uninitialized[UA]
; NODATASEC-NEXT:  .extern global_int_external_uninitialized[UA]
@global_int_external_uninitialized = external global i32, align 4

; CHECK-NEXT:      .extern tls_global_int_external_uninitialized[UL]
; NODATASEC-NEXT:  .extern tls_global_int_external_uninitialized[UL]
@tls_global_int_external_uninitialized = external thread_local global i32, align 4


; double global variable, TLS/Non-TLS, common/external linkage

; CHECK-NEXT:      .comm   global_double_common_zero_initialized[RW],8,3
; NODATASEC-NEXT:  .comm   global_double_common_zero_initialized[RW],8,3
@global_double_common_zero_initialized = common global double 0.000000e+00, align 8

; CHECK-NEXT:      .comm   tls_global_double_common_zero_initialized[UL],8,3
; NODATASEC-NEXT:  .comm   tls_global_double_common_zero_initialized[UL],8,3
@tls_global_double_common_zero_initialized = common thread_local global double 0.000000e+00, align 8

; CHECK-NEXT:      .extern global_double_external_uninitialized[UA]
; NODATASEC-NEXT:  .extern global_double_external_uninitialized[UA]
@global_double_external_uninitialized = external global i64, align 8

; CHECK-NEXT:      .extern tls_global_double_external_uninitialized[UL]
; NODATASEC-NEXT:  .extern tls_global_double_external_uninitialized[UL]
@tls_global_double_external_uninitialized = external thread_local global i64, align 8


; Long long global variable, TLS/Non-TLS, local/weak linkage

; CHECK-NEXT:      .csect  global_long_long_internal_val_initialized[RW],3
; CHECK-NEXT:      .lglobl global_long_long_internal_val_initialized[RW]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  4, 0
; CHECK-NEXT:      .vbyte  4, 1
; NODATASEC-NEXT:  .csect  .data[RW],3
; NODATASEC-NEXT:  .lglobl global_long_long_internal_val_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:global_long_long_internal_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
; NODATASEC-NEXT:  .vbyte  4, 1
@global_long_long_internal_val_initialized = internal global i64 1, align 8

; CHECK-NEXT:      .csect  tls_global_long_long_internal_val_initialized[TL],3
; CHECK-NEXT:      .lglobl tls_global_long_long_internal_val_initialized[TL]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  4, 0
; CHECK-NEXT:      .vbyte  4, 1
; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .lglobl tls_global_long_long_internal_val_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:tls_global_long_long_internal_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
; NODATASEC-NEXT:  .vbyte  4, 1
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
; CHECK-NEXT:      .vbyte  4, 0
; CHECK-NEXT:      .vbyte  4, 1
; NODATASEC-NEXT:  .csect  .data[RW],3
; NODATASEC-NEXT:  .weak   global_long_long_weak_val_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:global_long_long_weak_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
; NODATASEC-NEXT:  .vbyte  4, 1
@global_long_long_weak_val_initialized = weak global i64 1, align 8

; CHECK-NEXT:      .csect  tls_global_long_long_weak_val_initialized[TL],3
; CHECK-NEXT:      .weak   tls_global_long_long_weak_val_initialized[TL]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  4, 0
; CHECK-NEXT:      .vbyte  4, 1
; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .weak   tls_global_long_long_weak_val_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:tls_global_long_long_weak_val_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
; NODATASEC-NEXT:  .vbyte  4, 1
@tls_global_long_long_weak_val_initialized = weak thread_local global i64 1, align 8

; CHECK-NEXT:      .csect  global_long_long_weak_zero_initialized[RW],3
; CHECK-NEXT:      .weak   global_long_long_weak_zero_initialized[RW]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  4, 0
; CHECK-NEXT:      .vbyte  4, 0
; NODATASEC-NEXT:  .csect  .data[RW],3
; NODATASEC-NEXT:  .weak   global_long_long_weak_zero_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:global_long_long_weak_zero_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
; NODATASEC-NEXT:  .vbyte  4, 0
@global_long_long_weak_zero_initialized = weak global i64 0, align 8

; CHECK-NEXT:      .csect  tls_global_long_long_weak_zero_initialized[TL],3
; CHECK-NEXT:      .weak   tls_global_long_long_weak_zero_initialized[TL]
; CHECK-NEXT:      .align  3
; CHECK-NEXT:      .vbyte  4, 0
; CHECK-NEXT:      .vbyte  4, 0
; NODATASEC-NEXT:  .csect  .tdata[TL],3
; NODATASEC-NEXT:  .weak   tls_global_long_long_weak_zero_initialized
; NODATASEC-NEXT:  .align  3
; NODATASEC-NEXT:tls_global_long_long_weak_zero_initialized:
; NODATASEC-NEXT:  .vbyte  4, 0
; NODATASEC-NEXT:  .vbyte  4, 0
@tls_global_long_long_weak_zero_initialized = weak thread_local global i64 0, align 8
