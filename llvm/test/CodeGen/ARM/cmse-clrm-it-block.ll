; RUN: llc -mtriple=thumbv8.1m.main -mattr=+8msecext %s -o - | FileCheck %s

define hidden i32 @f(i32 %0, i32 (i32)* nocapture %1) local_unnamed_addr #0 {
  %3 = call i32 %1(i32 %0) #2
  %4 = icmp eq i32 %3, 1
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  call void bitcast (void (...)* @g to void ()*)() #3
  unreachable

6:                                                ; preds = %2
  ret i32 1
}
; CHECK-NOT: clrm eq
; CHECK: clrm {r1, r2, r3, r12, apsr}
; CHECK: bl g

declare dso_local void @g(...) local_unnamed_addr #1

attributes #0 = { nounwind "cmse_nonsecure_entry" }
attributes #1 = { noreturn }
attributes #2 = { nounwind "cmse_nonsecure_call" }
attributes #3 = { noreturn nounwind }
