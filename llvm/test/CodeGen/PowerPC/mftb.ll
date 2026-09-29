; Check handling of the mftb instruction.
; For CPUs 601 and pwr3, the mftb instruction should be emitted.
; On all other CPUs (including generic, ppc, ppc64), the mfspr instruction 
; should be used instead. There should no longer be a deprecated warning 
; message emittedfor this instruction for any CPU.

; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s 2>&1 \
; RUN:    | FileCheck %s --check-prefix=CHECK-MFSPR
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s 2>&1 \
; RUN:    | FileCheck %s --check-prefix=CHECK-MFSPR
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s 2>&1 \
; RUN:    | FileCheck %s --check-prefix=CHECK-MFSPR
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu  < %s 2>&1 \
; RUN:    | FileCheck %s --check-prefix=CHECK-MFSPR
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu -mcpu=ppc < %s 2>&1 \
; RUN:    | FileCheck %s --check-prefix=CHECK-MFSPR
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu -mcpu=601 < %s 2>&1 \
; RUN:    | FileCheck %s --check-prefix=CHECK-MFTB
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr3 < %s 2>&1 \
; RUN:    | FileCheck %s --check-prefix=CHECK-MFTB

; CHECK-MFSPR-NOT: warning: deprecated
; CHECK-MFTB-NOT: warning: deprecated

define i32 @get_time() {
       %time = call i32 asm "mftb $0, 268", "=r"()
       ret i32 %time
; CHECK-MFSPR-LABEL: @get_time
; CHECK-MFSPR: mfspr 3, 268
; CHECK-MFSPR: blr

; CHECK-MFTB-LABEL: @get_time
; CHECK-MFTB: mftb 3, 268
; CHECK-MFTB: blr
}

define i32 @get_timeu() {
       %time = call i32 asm "mftb $0, 269", "=r"()
       ret i32 %time
; CHECK-MFSPR-LABEL: @get_timeu
; CHECK-MFSPR: mfspr 3, 269
; CHECK-MFSPR: blr

; CHECK-MFTB-LABEL: @get_timeu
; CHECK-MFTB: mftbu 3
; CHECK-MFTB: blr
}

define i32 @get_time_e() {
       %time = call i32 asm "mftb $0", "=r"()
       ret i32 %time
; CHECK-MFSPR-LABEL: @get_time_e
; CHECK-MFSPR: mfspr 3, 268
; CHECK-MFSPR: blr

; CHECK-MFTB-LABEL: @get_time_e
; CHECK-MFTB: mftb 3, 268
; CHECK-MFTB: blr
}

define i32 @get_timeu_e() {
       %time = call i32 asm "mftbu $0", "=r"()
       ret i32 %time
; CHECK-MFSPR-LABEL: @get_timeu_e
; CHECK-MFSPR: mfspr 3, 269
; CHECK-MFSPR: blr

; CHECK-MFTB-LABEL: @get_timeu_e
; CHECK-MFTB: mftbu 3
; CHECK-MFTB: blr
}

