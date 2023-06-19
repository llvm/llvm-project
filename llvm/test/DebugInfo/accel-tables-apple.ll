; Verify the emission of accelerator tables for nameTableKind: Apple
; REQUIRES: x86-registered-target

; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj < %S/Inputs/name-table-kind-apple-5.ll \
; RUN:   | llvm-readobj --sections - | FileCheck --check-prefix=DEBUG_NAMES %s

; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj < %S/Inputs/name-table-kind-apple-4.ll \
; RUN:   | llvm-readobj --sections - | FileCheck --check-prefix=APPLE %s

; APPLE-NOT: debug_names
; APPLE-NOT: debug{{.*}}pub
; APPLE: apple_names
; APPLE-NOT: debug_names
; APPLE-NOT: debug{{.*}}pub

; DEBUG_NAMES-NOT: apple_names
; DEBUG_NAMES-NOT: pubnames
; DEBUG_NAMES: debug_names
; DEBUG_NAMES-NOT: apple_names
; DEBUG_NAMES-NOT: pubnames
