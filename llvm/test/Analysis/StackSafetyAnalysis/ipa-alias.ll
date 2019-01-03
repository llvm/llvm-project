; Test IPA over a single combined file
; RUN: llvm-as %s -o %t0.bc
; RUN: llvm-as %S/Inputs/ipa-alias.ll -o %t1.bc
; RUN: llvm-link %t0.bc %t1.bc -o %t.combined.bc
; RUN: opt -S -analyze -stack-safety-local %t.combined.bc | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -passes="print<stack-safety-local>" -disable-output %t.combined.bc 2>&1 | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -analyze -stack-safety %t.combined.bc | FileCheck %s --check-prefixes=CHECK,GLOBAL
; RUN: opt -S -passes="print-stack-safety" -disable-output %t.combined.bc 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @PreemptableAliasWrite1(i8* %p)
declare void @AliasToPreemptableAliasWrite1(i8* %p)

declare void @InterposableAliasWrite1(i8* %p)
; Aliases to interposable aliases are not allowed

declare void @AliasWrite1(i8* %p)

declare void @BitcastAliasWrite1(i32* %p)
declare void @AliasToBitcastAliasWrite1(i8* %p)

; Call to dso_preemptable alias to a dso_local aliasee
define void @PreemptableAliasCall() {
; CHECK-LABEL: @PreemptableAliasCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x1[1]: empty-set, @PreemptableAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x1[1]: full-set, @PreemptableAliasWrite1(arg0, [0,1)){{$}}
; LOCAL-NEXT: x2[1]: empty-set, @AliasToPreemptableAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x2[1]: [0,1), @AliasToPreemptableAliasWrite1(arg0, [0,1)){{$}}
; CHECK-NOT: ]:
entry:
  %x1 = alloca i8
  call void @PreemptableAliasWrite1(i8* %x1)

  %x2 = alloca i8
; Alias to a preemptable alias is not preemptable
  call void @AliasToPreemptableAliasWrite1(i8* %x2)
  ret void
}

; Call to an interposable alias to a non-interposable aliasee
define void @InterposableAliasCall() {
; CHECK-LABEL: @InterposableAliasCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @InterposableAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[1]: full-set, @InterposableAliasWrite1(arg0, [0,1)){{$}}
; CHECK-NOT: ]:
entry:
  %x = alloca i8
; ThinLTO can resolve the prevailing implementation for interposable definitions.
  call void @InterposableAliasWrite1(i8* %x)
  ret void
}

; Call to a dso_local/non-interposable alias/aliasee
define void @AliasCall() {
; CHECK-LABEL: @AliasCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @AliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[1]: [0,1), @AliasWrite1(arg0, [0,1)){{$}}
; CHECK-NOT: ]:
entry:
  %x = alloca i8
  call void @AliasWrite1(i8* %x)
  ret void
}

; Call to a bitcasted dso_local/non-interposable alias/aliasee
define void @BitcastAliasCall() {
; CHECK-LABEL: @BitcastAliasCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x1[4]: empty-set, @BitcastAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x1[4]: [0,1), @BitcastAliasWrite1(arg0, [0,1)){{$}}
; LOCAL-NEXT: x2[1]: empty-set, @AliasToBitcastAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x2[1]: [0,1), @AliasToBitcastAliasWrite1(arg0, [0,1)){{$}}
; CHECK-NOT: ]:
entry:
  %x1 = alloca i32
  call void @BitcastAliasWrite1(i32* %x1)
  %x2 = alloca i8
  call void @AliasToBitcastAliasWrite1(i8* %x2)
  ret void
}

; The rest is from Inputs/ipa-alias.ll

; CHECK-LABEL: @Write1{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NOT: ]:

; GLOBAL-LABEL: @InterposableAliasWrite1 interposable{{$}}
; GLOBAL-NEXT: args uses:
; GLOBAL-NEXT: <N/A>[]: [0,1), @Write1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: allocas uses:
; GLOBAL-NOT: ]:

; GLOBAL-LABEL: @PreemptableAliasWrite1 dso_preemptable{{$}}
; GLOBAL-NEXT: args uses:
; GLOBAL-NEXT: <N/A>[]: [0,1), @Write1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: allocas uses:
; GLOBAL-NOT: ]:

; GLOBAL-LABEL: @AliasToPreemptableAliasWrite1{{$}}
; GLOBAL-NEXT: args uses:
; GLOBAL-NEXT: <N/A>[]: [0,1), @Write1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: allocas uses:
; GLOBAL-NOT: ]:

; GLOBAL-LABEL: @AliasWrite1{{$}}
; GLOBAL-NEXT: args uses:
; GLOBAL-NEXT: <N/A>[]: [0,1), @Write1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: allocas uses:
; GLOBAL-NOT: ]:

; GLOBAL-LABEL: @BitcastAliasWrite1{{$}}
; GLOBAL-NEXT: args uses:
; GLOBAL-NEXT: <N/A>[]: [0,1), @Write1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: allocas uses:
; GLOBAL-NOT: ]:

; GLOBAL-LABEL: @AliasToBitcastAliasWrite1{{$}}
; GLOBAL-NEXT: args uses:
; GLOBAL-NEXT: <N/A>[]: [0,1), @Write1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: allocas uses:
; GLOBAL-NOT: ]:
