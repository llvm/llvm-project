; RUN: opt < %s -disable-output -passes='no-op-module' -print-before=bleh 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
; RUN: opt < %s -disable-output -passes='no-op-module' -print-after=bleh 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
; RUN: opt < %s -disable-output -passes='no-op-module' -print-before=no-op-function 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
; RUN: opt < %s -disable-output -passes='no-op-module' -print-after=no-op-function 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
; RUN: opt < %s -disable-output -passes='no-op-module,no-op-function' -print-before=no-op-module 2>&1 | FileCheck %s --check-prefix=ONCE
; RUN: opt < %s -disable-output -passes='no-op-module,no-op-function' -print-after=no-op-module 2>&1 | FileCheck %s --check-prefix=ONCE
; RUN: opt < %s -disable-output -passes='no-op-function' -print-before=no-op-function 2>&1 | FileCheck %s --check-prefix=ONCE
; RUN: opt < %s -disable-output -passes='no-op-function' -print-after=no-op-function 2>&1 | FileCheck %s --check-prefix=ONCE
; RUN: opt < %s -disable-output -passes='no-op-module,no-op-function' -print-before=no-op-function --print-module-scope 2>&1 | FileCheck %s --check-prefix=TWICE
; RUN: opt < %s -disable-output -passes='no-op-module,no-op-function' -print-after=no-op-function --print-module-scope 2>&1 | FileCheck %s --check-prefix=TWICE
; RUN: opt < %s -disable-output -passes='loop-vectorize' -print-before=loop-vectorize -print-after=loop-vectorize 2>&1 | FileCheck %s --check-prefix=CHECK-LV --allow-empty
; RUN: opt < %s -disable-output -passes='simple-loop-unswitch,simple-loop-unswitch<nontrivial>,simple-loop-unswitch<no-nontrivial>' -print-before=simple-loop-unswitch -print-after=simple-loop-unswitch 2>&1 | FileCheck %s --check-prefix=CHECK-UNSWITCH --allow-empty

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

; Verify that we can handle function passes with params.
; CHECK-LV: *** IR Dump Before LoopVectorizePass on foo ***
; CHECK-LV: *** IR Dump After LoopVectorizePass on foo ***
; CHECK-LV: *** IR Dump Before LoopVectorizePass on bar ***
; CHECK-LV: *** IR Dump After LoopVectorizePass on bar ***

; Verify that we can handle loop passes with params.
; CHECK-UNSWITCH: *** IR Dump Before SimpleLoopUnswitchPass on Loop at depth 1 containing
; CHECK-UNSWITCH: *** IR Dump After SimpleLoopUnswitchPass on Loop at depth 1 containing
; CHECK-UNSWITCH: *** IR Dump Before SimpleLoopUnswitchPass on Loop at depth 1 containing
; CHECK-UNSWITCH: *** IR Dump After SimpleLoopUnswitchPass on Loop at depth 1 containing
; CHECK-UNSWITCH: *** IR Dump Before SimpleLoopUnswitchPass on Loop at depth 1 containing
; CHECK-UNSWITCH: *** IR Dump After SimpleLoopUnswitchPass on Loop at depth 1 containing

define void @foo() {
  ret void
}

define void @bar() {
entry:
  br label %loop
loop:
  br label %loop
}
