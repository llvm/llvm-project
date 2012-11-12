; RUN: llc < %s -mtriple=i686-linux | FileCheck %s

@ok = internal constant [4 x i8] c"%d\0A\00"
@no = internal constant [4 x i8] c"no\0A\00"

define i1 @func1(i32 %v1, i32 %v2) nounwind {
entry:
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %normal

normal:
  %t1 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @ok, i32 0, i32 0), i32 %sum ) nounwind
  ret i1 true

overflow:
  %t2 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @no, i32 0, i32 0) ) nounwind
  ret i1 false

; CHECK: func1:
; CHECK: subl 20(%esp)
; CHECK-NEXT: jno
}

define i1 @func2(i32 %v1, i32 %v2) nounwind {
entry:
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %carry, label %normal

normal:
  %t1 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @ok, i32 0, i32 0), i32 %sum ) nounwind
  ret i1 true

carry:
  %t2 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @no, i32 0, i32 0) ) nounwind
  ret i1 false

; CHECK: func2:
; CHECK: subl 20(%esp)
; CHECK-NEXT: jae
}

declare i32 @printf(i8*, ...) nounwind
declare {i32, i1} @llvm.ssub.with.overflow.i32(i32, i32)
declare {i32, i1} @llvm.usub.with.overflow.i32(i32, i32)

define i1 @func3(i32 %x) nounwind {
entry:
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %x, i32 1)
  %obit = extractvalue {i32, i1} %t, 1
  ret i1 %obit

; CHECK: func3:
; CHECK: decl
; CHECK-NEXT: seto
}
