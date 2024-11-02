; RUN: llc < %s

; NVPTX fails to LowerFormalArguments for arg type i96
; the arg byte size must be one of the {16, 8, 4, 2}
; XFAIL: target=nvptx{{.*}}

@ok = internal constant [4 x i8] c"%d\0A\00"
@no = internal constant [4 x i8] c"no\0A\00"



define i1 @func2(i128 zeroext %v1, i128 zeroext %v2) nounwind {
entry:
  %t = call {i128, i1} @llvm.uadd.with.overflow.i128(i128 %v1, i128 %v2)
  %sum = extractvalue {i128, i1} %t, 0
  %sum32 = trunc i128 %sum to i32
  %obit = extractvalue {i128, i1} %t, 1
  br i1 %obit, label %carry, label %normal

normal:
  %t1 = tail call i32 (i8*, ...) @printf( i8* getelementptr ([4 x i8], [4 x i8]* @ok, i32 0, i32 0), i32 %sum32 ) nounwind
  ret i1 true

carry:
  %t2 = tail call i32 (i8*, ...) @printf( i8* getelementptr ([4 x i8], [4 x i8]* @no, i32 0, i32 0) ) nounwind
  ret i1 false
}

declare i32 @printf(i8*, ...) nounwind
declare {i96, i1} @llvm.sadd.with.overflow.i96(i96, i96)
declare {i128, i1} @llvm.uadd.with.overflow.i128(i128, i128)

define i1 @func1(i96 signext %v1, i96 signext %v2) nounwind {
entry:
  %t = call {i96, i1} @llvm.sadd.with.overflow.i96(i96 %v1, i96 %v2)
  %obit = extractvalue {i96, i1} %t, 1
  ret i1 %obit
}
