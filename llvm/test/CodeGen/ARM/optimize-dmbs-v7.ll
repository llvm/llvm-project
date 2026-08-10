; RUN: llc -O1 < %s -mtriple=armv7 -mattr=+db | FileCheck %s

@x1 = global i32 0, align 4
@x2 = global i32 0, align 4

define void @test() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.013 = phi i32 [ 1, %entry ], [ %inc6, %for.body ]
  store atomic i32 %i.013, ptr @x1 seq_cst, align 4
  store atomic i32 %i.013, ptr @x1 seq_cst, align 4
  store atomic i32 %i.013, ptr @x2 seq_cst, align 4
  %inc6 = add nsw i32 %i.013, 1
  %exitcond = icmp eq i32 %inc6, 2
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void

; The for.body contains 3 seq_cst stores.
; Hence it should have 3 dmb;str;dmb sequences with the middle dmbs collapsed
; CHECK: %for.body
; CHECK-NOT: str
; CHECK: dmb
; CHECK-NOT: dmb
; CHECK: str

; CHECK-NOT: str
; CHECK: dmb
; CHECK-NOT: dmb
; CHECK: str

; CHECK-NOT: str
; CHECK: dmb
; CHECK-NOT: dmb
; CHECK: str

; CHECK-NOT: str
; CHECK: dmb
; CHECK-NOT: dmb
; CHECK-NOT: str
; CHECK: %for.end
}

define void @test2() {
  call void @llvm.arm.dmb(i32 11)
  tail call void @test()
  call void @llvm.arm.dmb(i32 11)
  ret void
; the call should prevent the two dmbs from collapsing
; CHECK: test2:
; CHECK: dmb
; CHECK-NEXT: bl
; CHECK-NEXT: dmb
}

define void @test3() {
  call void @llvm.arm.dmb(i32 11)
  call void @llvm.arm.dsb(i32 9)
  call void @llvm.arm.dmb(i32 11)
  ret void
; the call should prevent the two dmbs from collapsing
; CHECK: test3:
; CHECK: dmb
; CHECK-NEXT: dsb
; CHECK-NEXT: dmb

}


declare void @llvm.arm.dmb(i32)
declare void @llvm.arm.dsb(i32)
