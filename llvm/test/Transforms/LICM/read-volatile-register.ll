; RUN: opt -S -licm %s | FileCheck %s

; Volatile register shouldn't be hoisted ourside loops.
define i32 @test_read() {
; CHECK-LABEL: define i32 @test_read()
; CHECK:     br label %loop
; CHECK: loop:
; CHECK:     %counter = tail call i64 @llvm.read_volatile_register

entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %inc ]
  %counter = tail call i64 @llvm.read_volatile_register.i64(metadata !1)
  %tst = icmp ult i64 %counter, 1000
  br i1 %tst, label %inc, label %done

inc:
  %i.next = add nuw nsw i32 %i, 1
  br label %loop

done:
  ret i32 %i
}

declare i64 @llvm.read_register.i64(metadata)
declare i64 @llvm.read_volatile_register.i64(metadata)

!1 = !{!"cntpct_el0"}
