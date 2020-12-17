; RUN: opt %s -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 | FileCheck %s

declare void @is(i1)

; If A == B is false then A == B is implied false.
; CHECK-LABEL: @test_eq_eq
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_eq_eq(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp eq i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A == B is false then A != B is implied true.
; CHECK-LABEL: @test_eq_ne
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_eq_ne(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ne i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A != B is false then A != B is implied false.
; CHECK-LABEL: @test_ne_ne
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_ne(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ne i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A != B is false then A >u B is implied false.
; CHECK-LABEL: @test_ne_ugt
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_ugt(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ugt i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A != B is false then A >=u B is implied true.
; CHECK-LABEL: @test_ne_uge
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_ne_uge(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp uge i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A != B is false then A <u B is implied false.
; CHECK-LABEL: @test_ne_ult
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ne_ult(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ult i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A != B is false then A <=u B is implied true.
; CHECK-LABEL: @test_ne_ule
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_ne_ule(i32 %a, i32 %b) {
  %cmp1 = icmp ne i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A >u B is false then A >u B is implied false.
; CHECK-LABEL: @test_ugt_ugt
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ugt_ugt(i32 %a, i32 %b) {
  %cmp1 = icmp ugt i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ugt i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A >u B is false then A <=u B is implied true.
; CHECK-LABEL: @test_ugt_ule
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_ugt_ule(i32 %a, i32 %b) {
  %cmp1 = icmp ugt i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A >=u B is false then A >=u B is implied false.
; CHECK-LABEL: @test_uge_uge
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_uge_uge(i32 %a, i32 %b) {
  %cmp1 = icmp uge i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp uge i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A >=u B is false then A <u B is implied true.
; CHECK-LABEL: @test_uge_ult
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_uge_ult(i32 %a, i32 %b) {
  %cmp1 = icmp uge i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ult i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A >=u B is false then A <=u B is implied true.
; CHECK-LABEL: @test_uge_ule
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_uge_ule(i32 %a, i32 %b) {
  %cmp1 = icmp uge i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A <u B is false then A <u B is implied false.
; CHECK-LABEL: @test_ult_ult
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ult_ult(i32 %a, i32 %b) {
  %cmp1 = icmp ult i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ult i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A <=u B is false then A <=u B is implied false.
; CHECK-LABEL: @test_ule_ule
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_ule_ule(i32 %a, i32 %b) {
  %cmp1 = icmp ule i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ule i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}
