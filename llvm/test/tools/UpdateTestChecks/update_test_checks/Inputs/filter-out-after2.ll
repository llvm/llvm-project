; RUN: opt < %s -S | FileCheck %s

define i32 @func1({i32, i32} %x, i32 %y, i1 %cond) {
entry:
  br i1 %cond, label %b1, label %b2

b1:
  %x.idx0 = extractvalue {i32, i32} %x, 0
  %add1 = add i32 %y, 1
  %add2 = add i32 %x.idx0, %add1
  %mul = mul i32 %add2, 3
  br label %b2

b2:
  %res = phi i32 [ -1, %entry ], [ %mul, %b1 ]
  ret i32 %res
}

define i32 @func2({i32, i32} %x, i32 %y, i1 %cond) {
entry:
  br i1 %cond, label %b2, label %b0

b2:
  %x.idx0 = extractvalue {i32, i32} %x, 0
  %add1 = add i32 %y, 1
  %add2 = add i32 %x.idx0, %add1
  %mul = mul i32 %add2, 3
  br label %b0

b0:
  %res = phi i32 [ -1, %entry ], [ %mul, %b2 ]
  ret i32 %res
}
