; RUN: opt < %s -passes=instcombine -S | not grep cmp
; rdar://6903175

define i1 @f0(ptr %a) nounwind {
       %b = load i32, ptr %a, align 4
       %c = uitofp i32 %b to double
       %d = fcmp ogt double %c, 0x41EFFFFFFFE00000
       ret i1 %d
}
