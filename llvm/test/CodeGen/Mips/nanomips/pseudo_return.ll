; RUN: llc -mtriple=nanomips -verify-machineinstrs --stop-after=finalize-isel < %s | FileCheck %s --check-prefix=AFTER-ISEL
; RUN: llc -mtriple=nanomips -verify-machineinstrs --stop-after=postrapseudos < %s | FileCheck %s --check-prefix=AFTER-POSTRA

define i32 @foo(i32 %a) {
; AFTER-ISEL: RetRA
; AFTER-POSTRA: PseudoReturnNM
  ret i32 %a
}
