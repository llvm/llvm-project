; RUN: llc -mtriple=arm-linux-gnueabi %s -o - | FileCheck %s

; Log2 and exp2 are string-matched to intrinsics. If they are not declared
; readnone, they can't be changed to intrinsics (because they can change errno).

declare double @log2(double)
declare double @exp2(double)

define void @f() {
       ; CHECK: bl log2
       %1 = call double @log2(double 0.000000e+00)
       ; CHECK: bl exp2
       %2 = call double @exp2(double 0.000000e+00)
       ret void
}
