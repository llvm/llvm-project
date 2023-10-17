; RUN: opt -passes="ipsccp<func-spec>" -force-specialization \
; RUN:   -funcspec-max-clones=2 -S < %s | FileCheck %s

; RUN: opt -passes="ipsccp<func-spec>" -force-specialization \
; RUN:   -funcspec-max-clones=1 -S < %s | FileCheck %s --check-prefix=CONST1

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
  %call = call i32 @foo(i32 %x, ptr @A, ptr @C)
  br label %return

if.else:
  %call1 = call i32 @foo(i32 %y, ptr @B, ptr @D)
  br label %return

return:
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
  ret i32 %retval.0
}

; CHECK-NOT: define internal i32 @foo(
define internal i32 @foo(i32 %x, ptr %b, ptr %c) {
entry:
  %0 = load i32, ptr %b, align 4
  %add = add nsw i32 %x, %0
  %1 = load i32, ptr %c, align 4
  %add1 = add nsw i32 %add, %1
  ret i32 %add1
}

; CONST1:     define internal i32 @foo.specialized.1(i32 %x, ptr %b, ptr %c)
; CONST1-NOT: define internal i32 @foo.specialized.2(i32 %x, ptr %b, ptr %c)

; CHECK:        define internal i32 @foo.specialized.1(i32 %x, ptr %b, ptr %c) {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %0 = load i32, ptr @A, align 4
; CHECK-NEXT:     %add = add nsw i32 %x, %0
; CHECK-NEXT:     %1 = load i32, ptr @C, align 4
; CHECK-NEXT:     %add1 = add nsw i32 %add, %1
; CHECK-NEXT:     ret i32 %add1
; CHECK-NEXT:   }

; CHECK: define internal i32 @foo.specialized.2(i32 %x, ptr %b, ptr %c) {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %0 = load i32, ptr @B, align 4
; CHECK-NEXT:     %add = add nsw i32 %x, %0
; CHECK-NEXT:     %1 = load i32, ptr @D, align 4
; CHECK-NEXT:     %add1 = add nsw i32 %add, %1
; CHECK-NEXT:     ret i32 %add1
; CHECK-NEXT:   }
