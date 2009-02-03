; RUN: llvm-as < %s | llc

@ok = internal constant [4 x i8] c"%d\0A\00"
@no = internal constant [4 x i8] c"no\0A\00"

define i1 @func1(i24 signext %v1, i24 signext %v2) nounwind {
entry:
  %t = call {i24, i1} @llvm.sadd.with.overflow.i24(i24 %v1, i24 %v2)
  %sum = extractvalue {i24, i1} %t, 0
  %sum32 = sext i24 %sum to i32
  %obit = extractvalue {i24, i1} %t, 1
  br i1 %obit, label %overflow, label %normal

normal:
  %t1 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @ok, i32 0, i32 0), i32 %sum32 ) nounwind
  ret i1 true

overflow:
  %t2 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @no, i32 0, i32 0) ) nounwind
  ret i1 false
}

define i1 @func2(i24 zeroext %v1, i24 zeroext %v2) nounwind {
entry:
  %t = call {i24, i1} @llvm.uadd.with.overflow.i24(i24 %v1, i24 %v2)
  %sum = extractvalue {i24, i1} %t, 0
  %sum32 = zext i24 %sum to i32
  %obit = extractvalue {i24, i1} %t, 1
  br i1 %obit, label %carry, label %normal

normal:
  %t1 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @ok, i32 0, i32 0), i32 %sum32 ) nounwind
  ret i1 true

carry:
  %t2 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @no, i32 0, i32 0) ) nounwind
  ret i1 false
}

declare i32 @printf(i8*, ...) nounwind
declare {i24, i1} @llvm.sadd.with.overflow.i24(i24, i24)
declare {i24, i1} @llvm.uadd.with.overflow.i24(i24, i24)
