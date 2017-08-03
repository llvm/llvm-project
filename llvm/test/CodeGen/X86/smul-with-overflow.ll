; RUN: llc < %s -mtriple=i686-- | FileCheck %s

@ok = internal constant [4 x i8] c"%d\0A\00"
@no = internal constant [4 x i8] c"no\0A\00"

define i1 @test1(i32 %v1, i32 %v2) nounwind {
entry:
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %normal

normal:
  %t1 = tail call i32 (i8*, ...) @printf( i8* getelementptr ([4 x i8], [4 x i8]* @ok, i32 0, i32 0), i32 %sum ) nounwind
  ret i1 true

overflow:
  %t2 = tail call i32 (i8*, ...) @printf( i8* getelementptr ([4 x i8], [4 x i8]* @no, i32 0, i32 0) ) nounwind
  ret i1 false
; CHECK-LABEL: test1:
; CHECK: imull
; CHECK-NEXT: jno
}

define i1 @test2(i32 %v1, i32 %v2) nounwind {
entry:
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %normal

overflow:
  %t2 = tail call i32 (i8*, ...) @printf( i8* getelementptr ([4 x i8], [4 x i8]* @no, i32 0, i32 0) ) nounwind
  ret i1 false

normal:
  %t1 = tail call i32 (i8*, ...) @printf( i8* getelementptr ([4 x i8], [4 x i8]* @ok, i32 0, i32 0), i32 %sum ) nounwind
  ret i1 true
; CHECK-LABEL: test2:
; CHECK: imull
; CHECK-NEXT: jno
}

declare i32 @printf(i8*, ...) nounwind
declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32)

define i32 @test3(i32 %a, i32 %b) nounwind readnone {
entry:
	%tmp0 = add i32 %b, %a
	%tmp1 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %tmp0, i32 2)
	%tmp2 = extractvalue { i32, i1 } %tmp1, 0
	ret i32 %tmp2
; CHECK-LABEL: test3:
; CHECK: addl
; CHECK-NEXT: addl
; CHECK-NEXT: ret
}

define i32 @test4(i32 %a, i32 %b) nounwind readnone {
entry:
	%tmp0 = add i32 %b, %a
	%tmp1 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %tmp0, i32 4)
	%tmp2 = extractvalue { i32, i1 } %tmp1, 0
	ret i32 %tmp2
; CHECK-LABEL: test4:
; CHECK: addl
; CHECK: mull
; CHECK-NEXT: ret
}

declare { i63, i1 } @llvm.smul.with.overflow.i63(i63, i63) nounwind readnone

define i1 @test5() nounwind {
entry:
  %res = call { i63, i1 } @llvm.smul.with.overflow.i63(i63 4, i63 4611686018427387903)
  %sum = extractvalue { i63, i1 } %res, 0
  %overflow = extractvalue { i63, i1 } %res, 1
  ret i1 %overflow
; Was returning false, should return true (not constant folded yet though).
; PR13991
; CHECK-LABEL: test5:
; CHECK-NOT: xorb
}
