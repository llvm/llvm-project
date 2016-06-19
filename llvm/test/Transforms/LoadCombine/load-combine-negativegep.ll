; RUN: opt -basicaa -load-combine -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @Load_NegGep(i32* %i){
  %1 = getelementptr inbounds i32, i32* %i, i64 -1
  %2 = load i32, i32* %1, align 4
  %3 = load i32, i32* %i, align 4
  %4 = add nsw i32 %3, %2
  ret i32 %4
; CHECK-LABEL: @Load_NegGep(
; CHECK:  %[[load:.*]] = load i64
; CHECK:  %[[combine_extract_lo:.*]] = trunc i64 %[[load]] to i32
; CHECK:  %[[combine_extract_shift:.*]] = lshr i64 %[[load]], 32
; CHECK:  %[[combine_extract_hi:.*]] = trunc i64 %[[combine_extract_shift]] to i32
; CHECK:  %[[add:.*]] = add nsw i32 %[[combine_extract_hi]], %[[combine_extract_lo]]
}


