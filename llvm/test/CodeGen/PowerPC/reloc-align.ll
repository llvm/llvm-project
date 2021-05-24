; RUN: llc -verify-machineinstrs -mcpu=pwr7 -O1 < %s | FileCheck %s

; This test verifies that the peephole optimization of address accesses
; does not produce a load or store with a relocation that can't be
; satisfied for a given instruction encoding.  Reduced from a test supplied
; by Hal Finkel.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.S1 = type { [8 x i8] }

@main.l_1554 = internal global { i8, i8, i8, i8, i8, i8, i8, i8 } { i8 -1, i8 -6, i8 57, i8 62, i8 -48, i8 0, i8 58, i8 80 }, align 1

; Function Attrs: nounwind readonly
define signext i32 @main() #0 {
entry:
  %call = tail call fastcc signext i32 @func_90(%struct.S1* byval(%struct.S1) bitcast ({ i8, i8, i8, i8, i8, i8, i8, i8 }* @main.l_1554 to %struct.S1*))
; CHECK-NOT: ld {{[0-9]+}}, main.l_1554@toc@l
  ret i32 %call
}

; Function Attrs: nounwind readonly
define internal fastcc signext i32 @func_90(%struct.S1* byval(%struct.S1) nocapture %p_91) #0 {
entry:
  %0 = bitcast %struct.S1* %p_91 to i64*
  %bf.load = load i64, i64* %0, align 1
  %bf.shl = shl i64 %bf.load, 26
  %bf.ashr = ashr i64 %bf.shl, 54
  %bf.cast = trunc i64 %bf.ashr to i32
  ret i32 %bf.cast
}

attributes #0 = { nounwind readonly "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
