; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.arm.cdp(i32, i32, i32, i32, i32, i32) nounwind

define void @cdp(i32 %a) #0 {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: %load = load i32, ptr %a.addr, align 4
  ; CHECK-NEXT: call void @llvm.arm.cdp(i32 %load, i32 2, i32 3, i32 4, i32 5, i32 6)
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %load = load i32, ptr %a.addr, align 4
  call void @llvm.arm.cdp(i32 %load, i32 2, i32 3, i32 4, i32 5, i32 6)
  ret void
}

declare void @llvm.arm.cdp2(i32, i32, i32, i32, i32, i32) nounwind
define void @cdp2(i32 %a) #0 {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: %load = load i32, ptr %a.addr, align 4
  ; CHECK-NEXT: call void @llvm.arm.cdp2(i32 %load, i32 2, i32 3, i32 4, i32 5, i32 6)
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %load = load i32, ptr %a.addr, align 4
  call void @llvm.arm.cdp2(i32 %load, i32 2, i32 3, i32 4, i32 5, i32 6)
  ret void
}

declare { i32, i32 } @llvm.arm.mrrc(i32, i32, i32) nounwind
define void @mrrc(i32 %arg0, i32 %arg1, i32 %arg2) #0 {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %ret0 = call { i32, i32 } @llvm.arm.mrrc(i32 %arg0, i32 0, i32 0)
  %ret0 = call { i32, i32 } @llvm.arm.mrrc(i32 %arg0, i32 0, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: %ret1 = call { i32, i32 } @llvm.arm.mrrc(i32 0, i32 %arg1, i32 0)
  %ret1 = call { i32, i32 } @llvm.arm.mrrc(i32 0, i32 %arg1, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret2 = call { i32, i32 } @llvm.arm.mrrc(i32 0, i32 0, i32 %arg2)
  %ret2 = call { i32, i32 } @llvm.arm.mrrc(i32 0, i32 0, i32 %arg2)
  ret void
}

declare { i32, i32 } @llvm.arm.mrrc2(i32, i32, i32) nounwind
define void @mrrc2(i32 %arg0, i32 %arg1, i32 %arg2) #0 {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: %ret0 = call { i32, i32 } @llvm.arm.mrrc2(i32 %arg0, i32 0, i32 0)
  %ret0 = call { i32, i32 } @llvm.arm.mrrc2(i32 %arg0, i32 0, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: %ret1 = call { i32, i32 } @llvm.arm.mrrc2(i32 0, i32 %arg1, i32 0)
  %ret1 = call { i32, i32 } @llvm.arm.mrrc2(i32 0, i32 %arg1, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret2 = call { i32, i32 } @llvm.arm.mrrc2(i32 0, i32 0, i32 %arg2)
  %ret2 = call { i32, i32 } @llvm.arm.mrrc2(i32 0, i32 0, i32 %arg2)
  ret void
}

declare void @llvm.arm.mcrr(i32, i32, i32, i32, i32) nounwind
define void @mcrr(i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: call void @llvm.arm.mcrr(i32 %arg0, i32 1, i32 2, i32 3, i32 4)
  call void @llvm.arm.mcrr(i32 %arg0, i32 1, i32 2, i32 3, i32 4)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: call void @llvm.arm.mcrr(i32 0, i32 %arg1, i32 2, i32 3, i32 4)
  call void @llvm.arm.mcrr(i32 0, i32 %arg1, i32 2, i32 3, i32 4)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg4
  ; CHECK-NEXT: call void @llvm.arm.mcrr(i32 0, i32 1, i32 2, i32 3, i32 %arg4)
  call void @llvm.arm.mcrr(i32 0, i32 1, i32 2, i32 3, i32 %arg4)
  ret void
}

declare void @llvm.arm.mcrr2(i32, i32, i32, i32, i32) nounwind
define void @mcrr2(i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: call void @llvm.arm.mcrr2(i32 %arg0, i32 1, i32 2, i32 3, i32 4)
  call void @llvm.arm.mcrr2(i32 %arg0, i32 1, i32 2, i32 3, i32 4)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: call void @llvm.arm.mcrr2(i32 0, i32 %arg1, i32 2, i32 3, i32 4)
  call void @llvm.arm.mcrr2(i32 0, i32 %arg1, i32 2, i32 3, i32 4)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg4
  ; CHECK-NEXT: call void @llvm.arm.mcrr2(i32 0, i32 1, i32 2, i32 3, i32 %arg4)
  call void @llvm.arm.mcrr2(i32 0, i32 1, i32 2, i32 3, i32 %arg4)
  ret void
}

declare i32 @llvm.arm.space(i32, i32) nounwind
define i32 @space(i32 %arg0, i32 %arg1) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg0
  ; CHECK-NEXT: call i32 @llvm.arm.space(i32 %arg0, i32 %arg1)
  %space = call i32 @llvm.arm.space(i32 %arg0, i32 %arg1)
  ret i32 %space
}
