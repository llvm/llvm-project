; RUN: not --crash opt < %s -disable-output -passes='no-op-module' -print-before=bleh 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
; RUN: not --crash opt < %s -disable-output -passes='no-op-module' -print-after=bleh 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
; RUN: opt < %s -disable-output -passes='no-op-module' -print-before=no-op-function 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
; RUN: opt < %s -disable-output -passes='no-op-module' -print-after=no-op-function 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
; RUN: opt < %s -disable-output -passes='no-op-module,no-op-function' -print-before=no-op-module 2>&1 | FileCheck %s --check-prefix=ONCE
; RUN: opt < %s -disable-output -passes='no-op-module,no-op-function' -print-after=no-op-module 2>&1 | FileCheck %s --check-prefix=ONCE
; RUN: opt < %s -disable-output -passes='no-op-function' -print-before=no-op-function 2>&1 | FileCheck %s --check-prefix=ONCE
; RUN: opt < %s -disable-output -passes='no-op-function' -print-after=no-op-function 2>&1 | FileCheck %s --check-prefix=ONCE
; RUN: opt < %s -disable-output -passes='no-op-module,no-op-function' -print-before=no-op-function --print-module-scope 2>&1 | FileCheck %s --check-prefix=TWICE
; RUN: opt < %s -disable-output -passes='no-op-module,no-op-function' -print-after=no-op-function --print-module-scope 2>&1 | FileCheck %s --check-prefix=TWICE

; NONE-NOT: @foo
; NONE-NOT: @bar

; ONCE: @foo
; ONCE: @bar
; ONCE-NOT: @foo
; ONCE-NOT: @bar

; TWICE: @foo
; TWICE: @bar
; TWICE: @foo
; TWICE: @bar
; TWICE-NOT: @foo
; TWICE-NOT: @bar

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}
