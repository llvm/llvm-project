; Check pass name is only printed once.
; Check only one function is printed
; RUN: opt < %s 2>&1 -forceattrs -disable-output -print-after-all -filter-print-funcs=foo | FileCheck %s  -check-prefix=FOO
; RUN: opt < %s 2>&1 -passes=forceattrs -disable-output -print-after-all -filter-print-funcs=foo | FileCheck %s  -check-prefix=FOO

; Check pass name is only printed once.
; Check both functions are printed
; RUN: opt < %s 2>&1 -forceattrs -disable-output -print-after-all -filter-print-funcs=foo,bar | FileCheck %s -check-prefix=BOTH
; RUN: opt < %s 2>&1 -passes=forceattrs -disable-output -print-after-all -filter-print-funcs=foo,bar | FileCheck %s -check-prefix=BOTH

; Check pass name is not printed if a module doesn't include any function specified in -filter-print-funcs.
; RUN: opt < %s 2>&1 -forceattrs -disable-output -print-after-all -filter-print-funcs=baz | FileCheck %s -allow-empty -check-prefix=EMPTY
; RUN: opt < %s 2>&1 -passes=forceattrs -disable-output -print-after-all -filter-print-funcs=baz | FileCheck %s -allow-empty -check-prefix=EMPTY

; Check whole module is printed with user-specified wildcast switch -filter-print-funcs=* or -print-module-scope
; RUN: opt < %s 2>&1 -forceattrs -disable-output -print-after-all | FileCheck %s -check-prefix=ALL
; RUN: opt < %s 2>&1 -forceattrs -disable-output  -print-after-all -filter-print-funcs=* | FileCheck %s -check-prefix=ALL
; RUN: opt < %s 2>&1 -forceattrs -disable-output -print-after-all -filter-print-funcs=foo -print-module-scope | FileCheck %s -check-prefix=ALL
; RUN: opt < %s 2>&1 -passes=forceattrs -disable-output -print-after-all | FileCheck %s -check-prefix=ALL
; RUN: opt < %s 2>&1 -passes=forceattrs -disable-output -print-after-all -filter-print-funcs=* | FileCheck %s -check-prefix=ALL
; RUN: opt < %s 2>&1 -passes=forceattrs -disable-output -print-after-all -filter-print-funcs=foo -print-module-scope | FileCheck %s -check-prefix=ALL

; FOO:      IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}
; FOO:      define void @foo
; FOO-NOT:  define void @bar
; FOO-NOT:  IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}

; BOTH:     IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}
; BOTH:     define void @foo
; BOTH:     define void @bar
; BOTH-NOT: IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}
; BOTH-NOT: ModuleID =

; EMPTY-NOT: IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}

; ALL:  IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}
; ALL:  ModuleID =
; ALL:  define void @foo
; ALL:  define void @bar
; ALL-NOT: IR Dump After {{Force set function attributes|ForceFunctionAttrsPass}}

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}
