; RUN: opt -S -passes='function(lower-expect),strip-dead-prototypes' < %s | FileCheck %s

; CHECK-LABEL: @test1(
define i32 @test1(i32 %x) nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %x.addr, align 4
  %cmp = icmp sgt i32 %tmp, 1
  %conv = zext i1 %cmp to i32
  %conv1 = sext i32 %conv to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv1, i64 1)
  %tobool = icmp ne i64 %expval, 0
; CHECK: !prof !0
; CHECK-NOT: @llvm.expect
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 (...) @f()
  store i32 %call, ptr %retval
  br label %return

if.end:                                           ; preds = %entry
  store i32 1, ptr %retval
  br label %return

return:                                           ; preds = %if.end, %if.then
  %0 = load i32, ptr %retval
  ret i32 %0
}

declare i64 @llvm.expect.i64(i64, i64) nounwind readnone

declare i32 @f(...)

; CHECK-LABEL: @test2(
define i32 @test2(i32 %x) nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %x.addr, align 4
  %conv = sext i32 %tmp to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 1)
  %tobool = icmp ne i64 %expval, 0
; CHECK: !prof !0
; CHECK-NOT: @llvm.expect
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 (...) @f()
  store i32 %call, ptr %retval
  br label %return

if.end:                                           ; preds = %entry
  store i32 1, ptr %retval
  br label %return

return:                                           ; preds = %if.end, %if.then
  %0 = load i32, ptr %retval
  ret i32 %0
}

; CHECK-LABEL: @test3(
define i32 @test3(i32 %x) nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %x.addr, align 4
  %tobool = icmp ne i32 %tmp, 0
  %lnot = xor i1 %tobool, true
  %lnot.ext = zext i1 %lnot to i32
  %conv = sext i32 %lnot.ext to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 1)
  %tobool1 = icmp ne i64 %expval, 0
; CHECK: !prof !0
; CHECK-NOT: @llvm.expect
  br i1 %tobool1, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 (...) @f()
  store i32 %call, ptr %retval
  br label %return

if.end:                                           ; preds = %entry
  store i32 1, ptr %retval
  br label %return

return:                                           ; preds = %if.end, %if.then
  %0 = load i32, ptr %retval
  ret i32 %0
}

; CHECK-LABEL: @test4(
define i32 @test4(i32 %x) nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %x.addr, align 4
  %tobool = icmp ne i32 %tmp, 0
  %lnot = xor i1 %tobool, true
  %lnot1 = xor i1 %lnot, true
  %lnot.ext = zext i1 %lnot1 to i32
  %conv = sext i32 %lnot.ext to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 1)
  %tobool2 = icmp ne i64 %expval, 0
; CHECK: !prof !0
; CHECK-NOT: @llvm.expect
  br i1 %tobool2, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 (...) @f()
  store i32 %call, ptr %retval
  br label %return

if.end:                                           ; preds = %entry
  store i32 1, ptr %retval
  br label %return

return:                                           ; preds = %if.end, %if.then
  %0 = load i32, ptr %retval
  ret i32 %0
}

; CHECK-LABEL: @test5(
define i32 @test5(i32 %x) nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %x.addr, align 4
  %cmp = icmp slt i32 %tmp, 0
  %conv = zext i1 %cmp to i32
  %conv1 = sext i32 %conv to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv1, i64 0)
  %tobool = icmp ne i64 %expval, 0
; CHECK: !prof !1
; CHECK-NOT: @llvm.expect
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 (...) @f()
  store i32 %call, ptr %retval
  br label %return

if.end:                                           ; preds = %entry
  store i32 1, ptr %retval
  br label %return

return:                                           ; preds = %if.end, %if.then
  %0 = load i32, ptr %retval
  ret i32 %0
}

; CHECK-LABEL: @test6(
define i32 @test6(i32 %x) nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %x.addr, align 4
  %conv = sext i32 %tmp to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 2)
; CHECK: !prof !2
; CHECK-NOT: @llvm.expect
  switch i64 %expval, label %sw.epilog [
    i64 1, label %sw.bb
    i64 2, label %sw.bb
  ]

sw.bb:                                            ; preds = %entry, %entry
  store i32 0, ptr %retval
  br label %return

sw.epilog:                                        ; preds = %entry
  store i32 1, ptr %retval
  br label %return

return:                                           ; preds = %sw.epilog, %sw.bb
  %0 = load i32, ptr %retval
  ret i32 %0
}

; CHECK-LABEL: @test7(
define i32 @test7(i32 %x) nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %x.addr, align 4
  %conv = sext i32 %tmp to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 1)
; CHECK: !prof !3
; CHECK-NOT: @llvm.expect
  switch i64 %expval, label %sw.epilog [
    i64 2, label %sw.bb
    i64 3, label %sw.bb
  ]

sw.bb:                                            ; preds = %entry, %entry
  %tmp1 = load i32, ptr %x.addr, align 4
  store i32 %tmp1, ptr %retval
  br label %return

sw.epilog:                                        ; preds = %entry
  store i32 0, ptr %retval
  br label %return

return:                                           ; preds = %sw.epilog, %sw.bb
  %0 = load i32, ptr %retval
  ret i32 %0
}

; CHECK-LABEL: @test8(
define i32 @test8(i32 %x) nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %x.addr, align 4
  %cmp = icmp sgt i32 %tmp, 1
  %conv = zext i1 %cmp to i32
  %expval = call i32 @llvm.expect.i32(i32 %conv, i32 1)
  %tobool = icmp ne i32 %expval, 0
; CHECK: !prof !0
; CHECK-NOT: @llvm.expect
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 (...) @f()
  store i32 %call, ptr %retval
  br label %return

if.end:                                           ; preds = %entry
  store i32 1, ptr %retval
  br label %return

return:                                           ; preds = %if.end, %if.then
  %0 = load i32, ptr %retval
  ret i32 %0
}

declare i32 @llvm.expect.i32(i32, i32) nounwind readnone

; CHECK-LABEL: @test9(
define i32 @test9(i32 %x) nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %x.addr, align 4
  %cmp = icmp sgt i32 %tmp, 1
  %expval = call i1 @llvm.expect.i1(i1 %cmp, i1 1)
; CHECK: !prof !0
; CHECK-NOT: @llvm.expect
  br i1 %expval, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 (...) @f()
  store i32 %call, ptr %retval
  br label %return

if.end:                                           ; preds = %entry
  store i32 1, ptr %retval
  br label %return

return:                                           ; preds = %if.end, %if.then
  %0 = load i32, ptr %retval
  ret i32 %0
}

; CHECK-LABEL: @test10(
define i32 @test10(i64 %t6) {
  %t7 = call i64 @llvm.expect.i64(i64 %t6, i64 0)
  %t8 = icmp ne i64 %t7, 0
  %t9 = select i1 %t8, i32 1, i32 2
; CHECK: select{{.*}}, !prof !1
  ret i32 %t9
}


declare i1 @llvm.expect.i1(i1, i1) nounwind readnone

; CHECK: !0 = !{!"branch_weights", !"expected", i32 2000, i32 1}
; CHECK: !1 = !{!"branch_weights", !"expected", i32 1, i32 2000}
; CHECK: !2 = !{!"branch_weights", !"expected", i32 1, i32 1, i32 2000}
; CHECK: !3 = !{!"branch_weights", !"expected", i32 2000, i32 1, i32 1}
