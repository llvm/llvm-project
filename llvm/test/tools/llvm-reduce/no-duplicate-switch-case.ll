; Ensure that llvm-reduce doesn't try to introduce a 0 or 1
; into a SwitchInst that already has one of those

; RUN: llvm-reduce --delta-passes=operands-zero --test %python --test-arg %p/Inputs/remove-bbs.py -abort-on-invalid-reduction %s -o %t

; RUN: llvm-reduce --delta-passes=operands-one --test %python --test-arg %p/Inputs/remove-bbs.py -abort-on-invalid-reduction %s -o %t

declare i32 @g()

define void @f(ptr %0, i1 %1) {
  %3 = alloca i32, align 4
  store ptr null, ptr %0, align 8
  %4 = call i32 @g()
  br i1 %1, label %5, label %7

5:                                                ; preds = %2
  br label %6

6:                                                ; preds = %5
  store i32 0, ptr %3, align 4
  br label %interesting2

7:                                                ; preds = %2
  br label %interesting2

interesting2:                                     ; preds = %7, %6
  %x9 = load i32, ptr %3, align 4
  switch i32 %x9, label %uninteresting [
    i32 3, label %interesting1
    i32 12, label %interesting1
  ]

interesting1:                                     ; preds = %8, %8
  ret void

uninteresting:                                    ; preds = %8
  unreachable
}
