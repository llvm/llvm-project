; RUN: opt -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -keep-loops=false -S < %s | FileCheck %s
; RUN: opt -passes='simplify-cfg<no-keep-loops>' -S < %s | FileCheck %s

define void @test1(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  %count = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = bitcast i32* %count to i8*
  store i32 0, i32* %count, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %entry
  %1 = load i32, i32* %count, align 4
  %2 = load i32, i32* %n.addr, align 4
  %cmp = icmp ule i32 %1, %2
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %3 = load i32, i32* %count, align 4
  %rem = urem i32 %3, 2
  %cmp1 = icmp eq i32 %rem, 0
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  %4 = load i32, i32* %count, align 4
  %add = add i32 %4, 1
  store i32 %add, i32* %count, align 4
  br label %if.end

; CHECK: if.then:
; CHECK:  br label %while.cond, !llvm.loop !0

if.else:                                          ; preds = %while.body
  %5 = load i32, i32* %count, align 4
  %add2 = add i32 %5, 2
  store i32 %add2, i32* %count, align 4
  br label %if.end

; CHECK: if.else:
; CHECK:  br label %while.cond, !llvm.loop !0

if.end:                                           ; preds = %if.else, %if.then
  br label %while.cond, !llvm.loop !0

while.end:                                        ; preds = %while.cond
  %6 = bitcast i32* %count to i8*
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.distribute.enable", i1 true}
; CHECK: !0 = distinct !{!0, !1}
; CHECK: !1 = !{!"llvm.loop.distribute.enable", i1 true}
