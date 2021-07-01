; RUN: opt -function-specialization -force-function-specialization \
; RUN:   -func-specialization-max-constants=2 -S < %s | FileCheck %s

; RUN: opt -function-specialization -force-function-specialization \
; RUN:   -func-specialization-max-constants=1 -S < %s | FileCheck %s --check-prefix=CONST1

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

@A = external dso_local constant i32, align 4
@B = external dso_local constant i32, align 4
@C = external dso_local constant i32, align 4
@D = external dso_local constant i32, align 4

define dso_local i32 @bar(i32 %x, i32 %y) {
entry:
  %tobool = icmp ne i32 %x, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
  %call = call i32 @foo(i32 %x, i32* @A, i32* @C)
  br label %return

if.else:
  %call1 = call i32 @foo(i32 %y, i32* @B, i32* @D)
  br label %return

return:
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
  ret i32 %retval.0
}

define internal i32 @foo(i32 %x, i32* %b, i32* %c) {
entry:
  %0 = load i32, i32* %b, align 4
  %add = add nsw i32 %x, %0
  %1 = load i32, i32* %c, align 4
  %add1 = add nsw i32 %add, %1
  ret i32 %add1
}

; CONST1-NOT: define internal i32 @foo.1(i32 %x, i32* %b, i32* %c)
; CONST1-NOT: define internal i32 @foo.2(i32 %x, i32* %b, i32* %c)

; CHECK:        define internal i32 @foo.1(i32 %x, i32* %b, i32* %c) {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %0 = load i32, i32* @A, align 4
; CHECK-NEXT:     %add = add nsw i32 %x, %0
; CHECK-NEXT:     %1 = load i32, i32* %c, align 4
; CHECK-NEXT:     %add1 = add nsw i32 %add, %1
; CHECK-NEXT:     ret i32 %add1
; CHECK-NEXT:   }

; CHECK: define internal i32 @foo.2(i32 %x, i32* %b, i32* %c) {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %0 = load i32, i32* @B, align 4
; CHECK-NEXT:     %add = add nsw i32 %x, %0
; CHECK-NEXT:     %1 = load i32, i32* %c, align 4
; CHECK-NEXT:     %add1 = add nsw i32 %add, %1
; CHECK-NEXT:     ret i32 %add1
; CHECK-NEXT:   }
