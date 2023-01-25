; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=ifuncs --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck --check-prefixes=CHECK-FINAL-IFUNCS,ALL --input-file=%t.0 %s

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=aliases,ifuncs --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t.1
; RUN: FileCheck --check-prefixes=CHECK-FINAL-BOTH,ALL --input-file=%t.1 %s

; Check interaction of reductions between aliases and ifuncs

; Test ifunc to alias
; CHECK-INTERESTINGNESS: @ifunc0_kept =


; ALL: [[TABLE:@[0-9]+]] = internal global [2 x ptr] poison
; ALL: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 10, ptr [[CONSTRUCTOR:@[0-9]+]], ptr null }]


; CHECK-FINAL-IFUNCS: @resolver_alias = alias ptr (), ptr @resolver
; CHECK-FINAL-IFUNCS: @ifunc_alias = alias ptr (), ptr @resolver_alias
; CHECK-FINAL-IFUNCS: @alias_of_ifunc = alias float (i64), ptr @ifunc_def

; CHECK-FINAL-IFUNCS: @ifunc0_kept = ifunc float (i64), ptr @resolver_alias
; CHECK-FINAL-IFUNCS: @ifunc_def = ifunc float (i64), ptr @resolver


; CHECK-FINAL-BOTH-NOT: _alias
; CHECK-FINAL-BOTH-NOT: @ifunc
; CHECK-FINAL-BOTH: @ifunc0_kept = ifunc float (i64), ptr @resolver
; CHECK-FINAL-BOTH-NOT: _alias
; CHECK-FINAL-BOTH-NOT: @ifunc
define ptr @resolver() {
  ret ptr inttoptr (i64 333 to ptr)
}

@resolver_alias = alias ptr (), ptr @resolver
@ifunc_alias = alias ptr (), ptr @resolver_alias

@ifunc0_kept = ifunc float (i64), ptr @resolver_alias
@ifunc1_removed = ifunc float (i64), ptr @resolver_alias

@ifunc_def = ifunc float (i64), ptr @resolver
@alias_of_ifunc = alias float (i64), ptr @ifunc_def

; ALL-LABEL: define float @call_ifunc_aliasee(i64 %arg) {
; ALL: %1 = load ptr, ptr [[TABLE]], align 8
; ALL: %call = call float %1(i64 %arg)
; ALL: ret float %call
define float @call_ifunc_aliasee(i64 %arg) {
  %call = call float @ifunc1_removed(i64 %arg)
  ret float %call
}

; ALL-LABEL: @call_alias_of_ifunc(
; CHECK-FINAL-IFUNCS: call float @alias_of_ifunc(

; CHECK-FINAL-BOTH-NEXT: %1 = load ptr, ptr getelementptr inbounds ([2 x ptr], ptr [[TABLE]], i32 0, i32 1), align 8
; CHECK-FINAL-BOTH-NEXT: %call = call float %1(i64 %arg)
; CHECK-FINAL-BOTH-NEXT: ret float %call
define float @call_alias_of_ifunc(i64 %arg) {
  %call = call float @alias_of_ifunc(i64 %arg)
  ret float %call
}

; CHECK-FINAL-BOTH: define internal void [[CONSTRUCTOR]]() {
; CHECK-FINAL-BOTH-NEXT: %1 = call ptr @resolver()
; CHECK-FINAL-BOTH-NEXT: store ptr %1, ptr [[TABLE]], align 8
; CHECK-FINAL-BOTH-NEXT: %2 = call ptr @resolver()
; CHECK-FINAL-BOTH-NEXT: store ptr %2, ptr getelementptr inbounds ([2 x ptr], ptr [[TABLE]], i32 0, i32 1), align 8
; CHECK-FINAL-BOTH-NEXT: ret void
