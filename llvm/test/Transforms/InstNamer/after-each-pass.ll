; RUN: opt -S -passes='function(instsimplify),strip' -instnamer-after-each-pass %s | FileCheck %s --check-prefix=FINAL
; RUN: opt -disable-output -passes='function(instsimplify),strip' -instnamer-after-each-pass -print-changed=diff %s 2>&1 | FileCheck %s --check-prefix=DIFF

; InstSimplify removes i.1. Strip removes the remaining names, so the
; instrumentation must assign fresh names before the next snapshot. In
; particular, it must not reuse i.1 for the surviving multiply.
define i32 @f(i32) {
entry:
  %1 = add i32 %0, 0
  %2 = mul i32 %1, 3
  ret i32 %2
}

; FINAL-LABEL: define i32 @f(i32 %arg.3)
; FINAL: bb.4:
; FINAL-NEXT: %i.5 = mul i32 %arg.3, 3
; FINAL-NEXT: ret i32 %i.5

; DIFF: *** IR Dump At Start ***
; DIFF: define i32 @f(i32 %arg.0)
; DIFF: %i.1 = add i32 %arg.0, 0
; DIFF: %i.2 = mul i32 %i.1, 3
; DIFF: *** IR Dump After InstSimplifyPass on f ***
; DIFF: *** IR Dump After StripSymbolsPass on [module] ***
