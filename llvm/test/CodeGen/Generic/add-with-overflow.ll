; RUN: llc < %s
; RUN: llc < %s -fast-isel

@ok = internal constant [4 x i8] c"%d\0A\00"
@no = internal constant [4 x i8] c"no\0A\00"

define i1 @func1(i32 %v1, i32 %v2) nounwind {
entry:
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %normal

normal:
  %t1 = tail call i32 (ptr, ...) @printf( ptr @ok, i32 %sum ) nounwind
  ret i1 true

overflow:
  %t2 = tail call i32 (ptr, ...) @printf( ptr @no ) nounwind
  ret i1 false
}

define i1 @func2(i32 %v1, i32 %v2) nounwind {
entry:
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %normal

normal:
  %t1 = tail call i32 (ptr, ...) @printf( ptr @ok, i32 %sum ) nounwind
  ret i1 true

overflow:
  %t2 = tail call i32 (ptr, ...) @printf( ptr @no ) nounwind
  ret i1 false
}

declare i32 @printf(ptr, ...) nounwind
declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)
declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32)
