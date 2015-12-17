; RUN: llc < %s -mtriple=x86_64-linux -enable-x86-lea-opt | FileCheck %s

%struct.anon1 = type { i32, i32, i32 }
%struct.anon2 = type { i32, [32 x i32], i32 }

@arr1 = external global [65 x %struct.anon1], align 16
@arr2 = external global [65 x %struct.anon2], align 16

define void @test1(i64 %x) nounwind {
entry:
  %a = getelementptr inbounds [65 x %struct.anon1], [65 x %struct.anon1]* @arr1, i64 0, i64 %x, i32 0
  %tmp = load i32, i32* %a, align 4
  %b = getelementptr inbounds [65 x %struct.anon1], [65 x %struct.anon1]* @arr1, i64 0, i64 %x, i32 1
  %tmp1 = load i32, i32* %b, align 4
  %sub = sub i32 %tmp, %tmp1
  %c = getelementptr inbounds [65 x %struct.anon1], [65 x %struct.anon1]* @arr1, i64 0, i64 %x, i32 2
  %tmp2 = load i32, i32* %c, align 4
  %add = add nsw i32 %sub, %tmp2
  switch i32 %add, label %sw.epilog [
    i32 1, label %sw.bb.1
    i32 2, label %sw.bb.2
  ]

sw.bb.1:                                          ; preds = %entry
  store i32 111, i32* %b, align 4
  store i32 222, i32* %c, align 4
  br label %sw.epilog

sw.bb.2:                                          ; preds = %entry
  store i32 333, i32* %b, align 4
  store i32 444, i32* %c, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb.2, %sw.bb.1, %entry
  ret void
; CHECK-LABEL: test1:
; CHECK:	leaq (%rdi,%rdi,2), [[REG1:%[a-z]+]]
; CHECK:	movl arr1(,[[REG1]],4), {{.*}}
; CHECK:	leaq arr1+4(,[[REG1]],4), [[REG2:%[a-z]+]]
; CHECK:	subl arr1+4(,[[REG1]],4), {{.*}}
; CHECK:	leaq arr1+8(,[[REG1]],4), [[REG3:%[a-z]+]]
; CHECK:	addl arr1+8(,[[REG1]],4), {{.*}}
; CHECK:	movl ${{[1-4]+}}, ([[REG2]])
; CHECK:	movl ${{[1-4]+}}, ([[REG3]])
; CHECK:	movl ${{[1-4]+}}, ([[REG2]])
; CHECK:	movl ${{[1-4]+}}, ([[REG3]])
}

define void @test2(i64 %x) nounwind optsize {
entry:
  %a = getelementptr inbounds [65 x %struct.anon1], [65 x %struct.anon1]* @arr1, i64 0, i64 %x, i32 0
  %tmp = load i32, i32* %a, align 4
  %b = getelementptr inbounds [65 x %struct.anon1], [65 x %struct.anon1]* @arr1, i64 0, i64 %x, i32 1
  %tmp1 = load i32, i32* %b, align 4
  %sub = sub i32 %tmp, %tmp1
  %c = getelementptr inbounds [65 x %struct.anon1], [65 x %struct.anon1]* @arr1, i64 0, i64 %x, i32 2
  %tmp2 = load i32, i32* %c, align 4
  %add = add nsw i32 %sub, %tmp2
  switch i32 %add, label %sw.epilog [
    i32 1, label %sw.bb.1
    i32 2, label %sw.bb.2
  ]

sw.bb.1:                                          ; preds = %entry
  store i32 111, i32* %b, align 4
  store i32 222, i32* %c, align 4
  br label %sw.epilog

sw.bb.2:                                          ; preds = %entry
  store i32 333, i32* %b, align 4
  store i32 444, i32* %c, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb.2, %sw.bb.1, %entry
  ret void
; CHECK-LABEL: test2:
; CHECK:	leaq (%rdi,%rdi,2), [[REG1:%[a-z]+]]
; CHECK:	leaq arr1+4(,[[REG1]],4), [[REG2:%[a-z]+]]
; CHECK:	movl -4([[REG2]]), {{.*}}
; CHECK:	subl ([[REG2]]), {{.*}}
; CHECK:	leaq arr1+8(,[[REG1]],4), [[REG3:%[a-z]+]]
; CHECK:	addl ([[REG3]]), {{.*}}
; CHECK:	movl ${{[1-4]+}}, ([[REG2]])
; CHECK:	movl ${{[1-4]+}}, ([[REG3]])
; CHECK:	movl ${{[1-4]+}}, ([[REG2]])
; CHECK:	movl ${{[1-4]+}}, ([[REG3]])
}

; Check that LEA optimization pass takes into account a resultant address
; displacement when choosing a LEA instruction for replacing a redundant
; address recalculation.

define void @test3(i64 %x) nounwind optsize {
entry:
  %a = getelementptr inbounds [65 x %struct.anon2], [65 x %struct.anon2]* @arr2, i64 0, i64 %x, i32 2
  %tmp = load i32, i32* %a, align 4
  %b = getelementptr inbounds [65 x %struct.anon2], [65 x %struct.anon2]* @arr2, i64 0, i64 %x, i32 0
  %tmp1 = load i32, i32* %b, align 4
  %add = add nsw i32 %tmp, %tmp1
  switch i32 %add, label %sw.epilog [
    i32 1, label %sw.bb.1
    i32 2, label %sw.bb.2
  ]

sw.bb.1:                                          ; preds = %entry
  store i32 111, i32* %a, align 4
  store i32 222, i32* %b, align 4
  br label %sw.epilog

sw.bb.2:                                          ; preds = %entry
  store i32 333, i32* %a, align 4
  store i32 444, i32* %b, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb.2, %sw.bb.1, %entry
  ret void
; CHECK-LABEL: test3:
; CHECK:	imulq {{.*}}, [[REG1:%[a-z]+]]
; CHECK:	leaq arr2+132([[REG1]]), [[REG2:%[a-z]+]]
; CHECK:	leaq arr2([[REG1]]), [[REG3:%[a-z]+]]

; REG3's definition is closer to movl than REG2's, but the pass still chooses
; REG2 because it provides the resultant address displacement fitting 1 byte.

; CHECK:	movl ([[REG2]]), {{.*}}
; CHECK:	addl ([[REG3]]), {{.*}}
; CHECK:	movl ${{[1-4]+}}, ([[REG2]])
; CHECK:	movl ${{[1-4]+}}, ([[REG3]])
; CHECK:	movl ${{[1-4]+}}, ([[REG2]])
; CHECK:	movl ${{[1-4]+}}, ([[REG3]])
}
