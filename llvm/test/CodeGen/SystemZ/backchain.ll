; Test the backchain attribute.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck -allow-deprecated-dag-overlap %s

declare i8 *@llvm.stacksave()
declare void @llvm.stackrestore(i8 *)
declare void @g()

; nothing should happen if no stack frame is needed.
define void @f1() "backchain" {
; CHECK-LABEL: f1:
; CHECK-NOT: stg
  ret void
}

; check that backchain is saved if we call someone
define void @f2() "backchain" {
; CHECK-LABEL: f2:
; CHECK: stmg %r14, %r15, 112(%r15)
; CHECK: lgr %r1, %r15
; CHECK: aghi %r15, -160
; CHECK: stg %r1, 0(%r15)
  call void @g()
  call void @g()
  ret void
}

; check that backchain is saved if we have an alloca
define void @f3() "backchain" {
; CHECK-LABEL: f3:
; CHECK-NOT: stmg
; CHECK: lgr %r1, %r15
; CHECK: aghi %r15, -168
; CHECK: stg %r1, 0(%r15)
  %ign = alloca i8, i32 4
  ret void
}

; check that alloca copies the backchain
define void @f4(i32 %len) "backchain" {
; CHECK-LABEL: f4:
; CHECK: stmg %r11, %r15, 88(%r15)
; CHECK: lgr %r1, %r15
; CHECK: aghi %r15, -160
; CHECK: stg %r1, 0(%r15)
; CHECK: lgr %r11, %r15
; CHECK-DAG: lg [[BC:%r[0-9]+]], 0(%r15)
; CHECK-DAG: lgr [[NEWSP:%r[0-9]+]], %r15
; CHECK: lgr %r15, [[NEWSP]]
; CHECK: stg [[BC]], 0([[NEWSP]])
  %ign = alloca i8, i32 %len
  ret void
}

; check that llvm.stackrestore restores the backchain
define void @f5(i32 %count1, i32 %count2) "backchain" {
; CHECK-LABEL: f5:
; CHECK: stmg %r11, %r15, 88(%r15)
; CHECK: lgr %r1, %r15
; CHECK: aghi %r15, -160
; CHECK: stg %r1, 0(%r15)
; CHECK: lgr %r11, %r15
; CHECK-DAG: lgr [[SAVESP:%r[0-9]+]], %r15
; CHECK-DAG: lg [[BC:%r[0-9]+]], 0(%r15)
; CHECK-DAG: lgr [[NEWSP:%r[0-9]+]], %r15
; CHECK-DAG: lgr %r15, [[NEWSP]]
; CHECK-DAG: stg [[BC]], 0([[NEWSP]])
; CHECK-DAG: lg [[BC2:%r[0-9]+]], 0(%r15)
; CHECK-DAG: lgr %r15, [[SAVESP]]
; CHECK-DAG: stg [[BC2]], 0([[SAVESP]])
; CHECK-DAG: lg [[BC3:%r[0-9]+]], 0(%r15)
; CHECK-DAG: lgr [[NEWSP2:%r[0-9]+]], %r15
; CHECK-DAG: lgr %r15, [[NEWSP2]]
; CHECK-DAG: stg [[BC3]], 0([[NEWSP2]])
; CHECK: lmg %r11, %r15, 248(%r11)
; CHECK: br %r14
  %src = call i8 *@llvm.stacksave()
  %array1 = alloca i8, i32 %count1
  store volatile i8 0, i8 *%array1
  call void @llvm.stackrestore(i8 *%src)
  %array2 = alloca i8, i32 %count2
  store volatile i8 0, i8 *%array2
  ret void
}

; same, but with the kernel backchain
define void @f6(i32 %count1, i32 %count2) #0 {
; CHECK-LABEL: f6:
; CHECK: stmg %r11, %r15, 112(%r15)
; CHECK: lgr %r1, %r15
; CHECK: aghi %r15, -48
; CHECK: stg %r1, 152(%r15)
; CHECK: lgr %r11, %r15
; CHECK-DAG: lgr [[SAVESP:%r[0-9]+]], %r15
; CHECK-DAG: lg [[BC:%r[0-9]+]], 152(%r15)
; CHECK-DAG: lgr [[NEWSP:%r[0-9]+]], %r15
; CHECK-DAG: lgr %r15, [[NEWSP]]
; CHECK-DAG: stg [[BC]], 152([[NEWSP]])
; CHECK-DAG: lg [[BC2:%r[0-9]+]], 152(%r15)
; CHECK-DAG: lgr %r15, [[SAVESP]]
; CHECK-DAG: stg [[BC2]], 152([[SAVESP]])
; CHECK-DAG: lg [[BC3:%r[0-9]+]], 152(%r15)
; CHECK-DAG: lgr [[NEWSP2:%r[0-9]+]], %r15
; CHECK-DAG: lgr %r15, [[NEWSP2]]
; CHECK-DAG: stg [[BC3]], 152([[NEWSP2]])
; CHECK: lmg %r11, %r15, 160(%r11)
; CHECK: br %r14
  %src = call i8 *@llvm.stacksave()
  %array1 = alloca i8, i32 %count1
  store volatile i8 0, i8 *%array1
  call void @llvm.stackrestore(i8 *%src)
  %array2 = alloca i8, i32 %count2
  store volatile i8 0, i8 *%array2
  ret void
}

attributes #0 = { "backchain" "packed-stack" "use-soft-float"="true" }
