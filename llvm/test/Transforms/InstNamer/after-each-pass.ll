; RUN: opt -S -passes='function(instsimplify),strip' -instnamer-after-each-pass %s | FileCheck %s --check-prefix=FINAL
; RUN: opt -disable-output -passes='function(instsimplify),strip' -instnamer-after-each-pass -print-changed %s 2>&1 | FileCheck %s --check-prefix=CHANGED

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

; CHANGED: *** IR Dump At Start ***
; CHANGED: define i32 @f(i32 %arg.0)
; CHANGED: %i.1 = add i32 %arg.0, 0
; CHANGED: %i.2 = mul i32 %i.1, 3
; CHANGED: *** IR Dump After InstSimplifyPass on f ***
; CHANGED: %i.2 = mul i32 %arg.0, 3
; CHANGED: *** IR Dump After StripSymbolsPass on [module] ***
; CHANGED: define i32 @f(i32 %arg.3)
; CHANGED: bb.4:
; CHANGED: %i.5 = mul i32 %arg.3, 3
