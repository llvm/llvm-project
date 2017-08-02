; RUN: llc < %s -mtriple=i386-- | grep "jc" | count 1
; XFAIL: *

; FIXME: umul-with-overflow not supported yet.

@ok = internal constant [4 x i8] c"%d\0A\00"
@no = internal constant [4 x i8] c"no\0A\00"

define i1 @func(i32 %v1, i32 %v2) nounwind {
entry:
  %t = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %carry, label %normal

normal:
  %t1 = tail call i32 (i8*, ...) @printf( i8* getelementptr ([4 x i8], [4 x i8]* @ok, i32 0, i32 0), i32 %sum ) nounwind
  ret i1 true

carry:
  %t2 = tail call i32 (i8*, ...) @printf( i8* getelementptr ([4 x i8], [4 x i8]* @no, i32 0, i32 0) ) nounwind
  ret i1 false
}

declare i32 @printf(i8*, ...) nounwind
declare {i32, i1} @llvm.umul.with.overflow.i32(i32, i32)
