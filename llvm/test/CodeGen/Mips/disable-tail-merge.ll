; RUN: llc -mtriple=mipsel < %s | FileCheck %s

@g0 = common global i32 0, align 4
@g1 = common global i32 0, align 4

; CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, 23
; CHECK: addiu ${{[0-9]+}}, ${{[0-9]+}}, 23

define i32 @test1(i32 %a) {
entry:
  %tobool = icmp eq i32 %a, 0
  %0 = load i32, ptr @g0, align 4
  br i1 %tobool, label %if.else, label %if.then

if.then:
  %add = add nsw i32 %0, 1
  store i32 %add, ptr @g0, align 4
  %1 = load i32, ptr @g1, align 4
  %add1 = add nsw i32 %1, 23
  br label %if.end

if.else:
  %add2 = add nsw i32 %0, 11
  store i32 %add2, ptr @g0, align 4
  %2 = load i32, ptr @g1, align 4
  %add3 = add nsw i32 %2, 23
  br label %if.end

if.end:
  %storemerge = phi i32 [ %add3, %if.else ], [ %add1, %if.then ]
  store i32 %storemerge, ptr @g1, align 4
  ret i32 %storemerge
}
