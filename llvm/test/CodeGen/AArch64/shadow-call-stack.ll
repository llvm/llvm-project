; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu -mattr=+reserve-x18 | FileCheck %s

define void @f1() shadowcallstack {
  ; CHECK: f1:
  ; CHECK-NOT: x18
  ; CHECK: ret
  ret void
}

declare void @foo()

define void @f2() shadowcallstack {
  ; CHECK: f2:
  ; CHECK-NOT: x18
  ; CHECK: b foo
  tail call void @foo()
  ret void
}

declare i32 @bar()

define i32 @f3() shadowcallstack {
  ; CHECK: f3:
  ; CHECK: str x30, [x18], #8
  ; CHECK: .cfi_escape 0x16, 0x12, 0x02, 0x82, 0x78
  ; CHECK: str x30, [sp, #-16]!
  %res = call i32 @bar()
  %res1 = add i32 %res, 1
  ; CHECK: ldr x30, [sp], #16
  ; CHECK: ldr x30, [x18, #-8]!
  ; CHECK: ret
  ret i32 %res
}

define i32 @f4() shadowcallstack {
  ; CHECK: f4:
  %res1 = call i32 @bar()
  %res2 = call i32 @bar()
  %res3 = call i32 @bar()
  %res4 = call i32 @bar()
  %res12 = add i32 %res1, %res2
  %res34 = add i32 %res3, %res4
  %res1234 = add i32 %res12, %res34
  ; CHECK: ldp {{.*}}x30, [sp
  ; CHECK: ldr x30, [x18, #-8]!
  ; CHECK: ret
  ret i32 %res1234
}

define i32 @f5() shadowcallstack nounwind {
  ; CHECK: f5:
  ; CHECK-NOT: .cfi_escape
  %res = call i32 @bar()
  %res1 = add i32 %res, 1
  ret i32 %res
}
