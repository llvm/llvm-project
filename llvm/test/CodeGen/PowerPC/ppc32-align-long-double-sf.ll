; RUN: llc -verify-machineinstrs -O2 -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

@x = global ppc_fp128 0xM405EDA5E353F7CEE0000000000000000, align 16
@.str = private unnamed_addr constant [5 x i8] c"%Lf\0A\00", align 1


define void @foo() #0 {
entry:
  %0 = load ppc_fp128, ptr @x, align 16
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, ppc_fp128 %0)
  ret void
}
; Do not skip register r4 because of register alignment in soft float mode. Instead skipping 
; put in r4 part of first argument for printf function (long double).
; CHECK: lwzu 4, x@l({{[0-9]+}})

declare i32 @printf(ptr nocapture readonly, ...) #0

attributes #0 = { "use-soft-float"="true" }

                        
