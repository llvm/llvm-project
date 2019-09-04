; Check pass name is only printed once.
; RUN: opt < %s 2>&1 -forceattrs -disable-output -print-after-all | FileCheck %s
; RUN: opt < %s 2>&1 -forceattrs -disable-output -print-after-all -filter-print-funcs=foo,bar | FileCheck %s

; Check pass name is not printed if a module doesn't include any function specified in -filter-print-funcs.
; RUN: opt < %s 2>&1 -forceattrs -disable-output -print-after-all -filter-print-funcs=baz | FileCheck %s -allow-empty -check-prefix=EMPTY

; CHECK: *** IR Dump After Force set function attributes ***
; CHECK-NOT: *** IR Dump After Force set function attributes ***
; EMPTY-NOT: *** IR Dump After Force set function attributes ***

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}
