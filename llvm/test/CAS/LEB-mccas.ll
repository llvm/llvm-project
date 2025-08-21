; This test was reduced using delta and llvm-reduce on a crash in MCCAS because
; of an FT_LEB fragment which was attempted to be merged into an 
; MCMergedFragmentRef. If MCCAS behaves correctly, llc should not crash when 
; trying to create an MCCAS representation of this LLVM IR file.

; RUN: rm -rf %t && mkdir -p %t
; RUN: llc --filetype=obj --mccas-verify --cas-backend --cas-friendly-debug-info --cas=%t/cas %s -o %t/LEB.o 

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: noinline optnone
define { ptr, i32 } @appendAnimation() #0 personality ptr @__objc_personality_v0 {
  %1 = invoke ptr null(ptr null, ptr null)
          to label %2 unwind label %3

2:                                                ; preds = %0
  ret { ptr, i32 } zeroinitializer

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          catch ptr null
  ret { ptr, i32 } %4
}

declare i32 @__objc_personality_v0(...)

attributes #0 = { noinline optnone }
