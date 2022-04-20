; REQUIRES: aarch64-registered-target
; REQUIRES: shell

; Test IPA over a single combined file
; RUN: llvm-as %s -o %t0.bc
; RUN: llvm-as %S/Inputs/ipa-alias.ll -o %t1.bc
; RUN: llvm-link %t0.bc %t1.bc -o %t.combined.bc

; RUN: opt -S -passes="print<stack-safety-local>" -disable-output %t.combined.bc 2>&1 | FileCheck %s --check-prefixes=CHECK,LOCAL

; RUN: opt -S -passes="print-stack-safety" -disable-output %t.combined.bc 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL,NOLTO

; Do an end-to-test using the new LTO API
; RUN: opt -module-summary %s -o %t.summ0.bc
; RUN: opt -module-summary %S/Inputs/ipa-alias.ll -o %t.summ1.bc

; RUN: echo > %t.res.txt \
; RUN:  -r %t.summ0.bc,AliasCall,px \
; RUN:  -r %t.summ0.bc,AliasToBitcastAliasWrite1, \
; RUN:  -r %t.summ0.bc,AliasToPreemptableAliasWrite1, \
; RUN:  -r %t.summ0.bc,AliasWrite1, \
; RUN:  -r %t.summ0.bc,BitcastAliasCall,px \
; RUN:  -r %t.summ0.bc,BitcastAliasWrite1, \
; RUN:  -r %t.summ0.bc,InterposableAliasCall,px \
; RUN:  -r %t.summ0.bc,InterposableAliasWrite1, \
; RUN:  -r %t.summ0.bc,PreemptableAliasCall,px \
; RUN:  -r %t.summ0.bc,PreemptableAliasWrite1, \
; RUN:  -r %t.summ1.bc,AliasToBitcastAliasWrite1,px \
; RUN:  -r %t.summ1.bc,AliasToPreemptableAliasWrite1,px \
; RUN:  -r %t.summ1.bc,AliasWrite1,px \
; RUN:  -r %t.summ1.bc,BitcastAliasWrite1,px \
; RUN:  -r %t.summ1.bc,InterposableAliasWrite1,px \
; RUN:  -r %t.summ1.bc,PreemptableAliasWrite1,px \
; RUN:  -r %t.summ1.bc,Write1,px

; RUN: llvm-lto2 run %t.summ0.bc %t.summ1.bc -o %t.lto -stack-safety-print -stack-safety-run -save-temps -thinlto-threads 1 -O0 \
; RUN:  $(cat %t.res.txt) \
; RUN:    2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL,LTO

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

attributes #0 = { noinline sanitize_memtag "target-features"="+mte,+neon" }

declare void @PreemptableAliasWrite1(i8* %p)
declare void @AliasToPreemptableAliasWrite1(i8* %p)

declare void @InterposableAliasWrite1(i8* %p)
; Aliases to interposable aliases are not allowed

declare void @AliasWrite1(i8* %p)

declare void @BitcastAliasWrite1(i32* %p)
declare void @AliasToBitcastAliasWrite1(i8* %p)

; Call to dso_preemptable alias to a dso_local aliasee
define void @PreemptableAliasCall() #0 {
; CHECK-LABEL: @PreemptableAliasCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x1[1]: empty-set, @PreemptableAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x1[1]: full-set, @PreemptableAliasWrite1(arg0, [0,1)){{$}}
; LOCAL-NEXT: x2[1]: empty-set, @AliasToPreemptableAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x2[1]: [0,1), @AliasToPreemptableAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x1 = alloca i8
  call void @PreemptableAliasWrite1(i8* %x1)

  %x2 = alloca i8
; Alias to a preemptable alias is not preemptable
  call void @AliasToPreemptableAliasWrite1(i8* %x2)
  ret void
}

; Call to an interposable alias to a non-interposable aliasee
define void @InterposableAliasCall() #0 {
; CHECK-LABEL: @InterposableAliasCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @InterposableAliasWrite1(arg0, [0,1)){{$}}
; NOLTO-NEXT: x[1]: full-set, @InterposableAliasWrite1(arg0, [0,1)){{$}}
; LTO-NEXT: x[1]: [0,1), @InterposableAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i8
; ThinLTO can resolve the prevailing implementation for interposable definitions.
  call void @InterposableAliasWrite1(i8* %x)
  ret void
}

; Call to a dso_local/non-interposable alias/aliasee
define void @AliasCall() #0 {
; CHECK-LABEL: @AliasCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @AliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[1]: [0,1), @AliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i8
  call void @AliasWrite1(i8* %x)
  ret void
}

; Call to a bitcasted dso_local/non-interposable alias/aliasee
define void @BitcastAliasCall() #0 {
; CHECK-LABEL: @BitcastAliasCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x1[4]: empty-set, @BitcastAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x1[4]: [0,1), @BitcastAliasWrite1(arg0, [0,1)){{$}}
; LOCAL-NEXT: x2[1]: empty-set, @AliasToBitcastAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x2[1]: [0,1), @AliasToBitcastAliasWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
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
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, i8* %p, align 1
; CHECK-EMPTY:
