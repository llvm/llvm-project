; RUN: llvm-reduce --delta-passes=basic-blocks --test %python --test-arg %p/Inputs/remove-bbs.py -abort-on-invalid-reduction %s -o %t

declare void @0()

define internal void @1(ptr %x0, i1 %x1) {
interesting0:
  %x3 = alloca i32, align 4
  store ptr null, ptr %x0, align 8
  br label %interesting1

interesting1:                                                ; preds = %x2
  call void @0()
  br label %x2

x2:
  br i1 %x1, label %interesting3, label %interesting4

interesting3:
  call void @0()
  br label %x2

interesting4:
  br label %x5

x5:
  store i32 0, ptr %x3, align 4
  ret void
}
