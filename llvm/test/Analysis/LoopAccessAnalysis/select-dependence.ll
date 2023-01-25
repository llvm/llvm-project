; RUN: opt -passes='print<access-info>' -disable-output 2>&1 < %s | FileCheck %s

; CHECK: Dependences:
; CHECK-NEXT: Unknown:
; CHECK-NEXT: %t63 = load double, ptr %t62, align 8 ->
; CHECK-NEXT: store double %t63, ptr %t64, align 8

define i32 @test() {
   %a1 = alloca [128 x double], align 8
   %a2 = alloca [128 x double], align 8
   %a3 = alloca [128 x double], align 8
   %t30 = getelementptr double, ptr %a2, i64 -32
   br label %loop

loop:
   %t58 = phi i64 [ %t65, %loop ], [ 0, %0 ]
   %t59 = icmp ule i64 %t58, 32
   %t60 = select i1 %t59, ptr %a1, ptr %t30
   %t62 = getelementptr inbounds double, ptr %t60, i64 %t58
   %t63 = load double, ptr %t62, align 8
   %t61 = select i1 %t59, ptr %a2, ptr %a3
   %t64 = getelementptr inbounds double, ptr %t61, i64 %t58
   store double %t63, ptr %t64, align 8
   %t65 = add nuw nsw i64 %t58, 1
   %t66 = icmp eq i64 %t65, 94
   br i1 %t66, label %exit, label %loop

exit:
   ret i32 0
}
