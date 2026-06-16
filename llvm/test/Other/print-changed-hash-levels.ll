; RUN: opt -S -passes=instsimplify -print-changed=quiet,hash \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s --check-prefix=FUNC
; RUN: opt -S -passes=instsimplify -print-changed=quiet,hash-func \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s --check-prefix=FUNC
; RUN: opt -S -passes=instsimplify -print-changed=quiet,hash-bb \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s --check-prefix=BB
; RUN: opt -S -passes=simplifycfg -print-changed=quiet,hash-bb \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s --check-prefix=SINGLE
; RUN: opt -S -passes="no-op-function,instsimplify,no-op-function" \
; RUN:   -filter-passes="no-op-function" -print-changed=hash \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s --check-prefix=FILTER
; RUN: not opt -S -passes=instsimplify \
; RUN:   -print-changed=hash-func,hash-bb -disable-output < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-HASH-MODE
; RUN: not opt -S -passes=instsimplify \
; RUN:   -print-changed=hash,diff -disable-output < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BAD-HASH-DIFF

define i32 @f(i1 %c, i32 %x) {
entry:
  br i1 %c, label %then, label %else

then:
  %a = add i32 %x, 0
  ret i32 %a

else:
  ret i32 %x
}

define i32 @collapse() {
entry:
  br label %exit

exit:
  ret i32 0
}

; FUNC: *** IR Function Changes After InstSimplifyPass on f ***
; FUNC-NEXT: *** Function f changed ***
; FUNC-NEXT: define i32 @f(i1 %c, i32 %x) {
; FUNC: ret i32 %x

; BB: *** IR BasicBlock Changes After InstSimplifyPass on f ***
; BB-NEXT: *** Function f changed ***
; BB-NEXT: *** BasicBlock f:then changed ***
; BB: then:
; BB: ret i32 %x
; BB: ; 2 unchanged basic blocks omitted

; SINGLE: *** IR BasicBlock Changes After SimplifyCFGPass on collapse ***
; SINGLE-NEXT: *** Function collapse changed ***
; SINGLE-NEXT: define i32 @collapse() {
; SINGLE: ret i32 0

; FILTER: *** IR Dump After NoOpFunctionPass on f omitted because no change ***
; FILTER: *** IR Dump After InstSimplifyPass on f filtered out ***
; FILTER-NOT: *** IR Function Changes After NoOpFunctionPass on f ***
; FILTER: *** IR Dump After NoOpFunctionPass on f omitted because no change ***

; BAD-HASH-MODE: LLVM ERROR: invalid argument 'hash-bb' to -print-changed=;
; BAD-HASH-DIFF: LLVM ERROR: invalid argument 'hash' to -print-changed=;
