; RUN: opt -S -passes='require<profile-summary>,function(codegenprepare)' -disable-complex-addr-modes=false -addr-sink-new-phis=true -addr-sink-new-select=true -disable-cgp-delete-phis  %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-YES
; RUN: opt -S -passes='require<profile-summary>,function(codegenprepare)' -disable-complex-addr-modes=false -addr-sink-new-phis=false -addr-sink-new-select=true -disable-cgp-delete-phis %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NO
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

; Can we sink for different base if there is no phi for base?
define i32 @test1(i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test1
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br label %fallthrough

fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO-LABEL: fallthrough:
; CHECK-NO: phi
; CHECK-NO-NEXT: load
  %c = phi ptr [%a1, %entry], [%a2, %if.then]
  %v = load i32, ptr %c, align 4
  ret i32 %v
}

; Can we sink for different base if there is phi for base?
define i32 @test2(i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test2
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br label %fallthrough

fallthrough:
; CHECK: getelementptr inbounds i8, {{.+}} 40
  %b = phi ptr [%b1, %entry], [%b2, %if.then]
  %c = phi ptr [%a1, %entry], [%a2, %if.then]
  %v = load i32, ptr %c, align 4
  ret i32 %v
}

; Can we sink for different base if there is phi for base but not valid one?
define i32 @test3(i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test3
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br label %fallthrough

fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO-LABEL: fallthrough:
; CHECK-NO: phi
; CHECK-NO: phi
; CHECK-NO-NEXT: load
  %b = phi ptr [%b2, %entry], [%b1, %if.then]
  %c = phi ptr [%a1, %entry], [%a2, %if.then]
  %v = load i32, ptr %c, align 4
  ret i32 %v
}

; Can we sink for different base if both addresses are in the same block?
define i32 @test4(i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test4
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  br label %fallthrough

fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO-LABEL: fallthrough:
; CHECK-NO: phi
; CHECK-NO-NEXT: load
  %c = phi ptr [%a1, %entry], [%a2, %if.then]
  %v = load i32, ptr %c, align 4
  ret i32 %v
}

; Can we sink for different base if there is phi for base?
; Both addresses are in the same block.
define i32 @test5(i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test5
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  br label %fallthrough

fallthrough:
; CHECK: getelementptr inbounds i8, {{.+}} 40
  %b = phi ptr [%b1, %entry], [%b2, %if.then]
  %c = phi ptr [%a1, %entry], [%a2, %if.then]
  %v = load i32, ptr %c, align 4
  ret i32 %v
}

; Can we sink for different base if there is phi for base but not valid one?
; Both addresses are in the same block.
define i32 @test6(i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test6
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  br label %fallthrough

fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO-LABEL: fallthrough:
; CHECK-NO: phi
; CHECK-NO-NEXT: phi
; CHECK-NO-NEXT: load
  %b = phi ptr [%b2, %entry], [%b1, %if.then]
  %c = phi ptr [%a1, %entry], [%a2, %if.then]
  %v = load i32, ptr %c, align 4
  ret i32 %v
}

; case with a loop. No phi node.
define i32 @test7(i32 %N, i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test7
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  br label %loop

loop:
; CHECK-LABEL: loop:
; CHECK-YES: sunk_phi
  %iv = phi i32 [0, %entry], [%iv.inc, %fallthrough]
  %c3 = phi ptr [%a1, %entry], [%c, %fallthrough]
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br label %fallthrough

fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO-LABEL: fallthrough:
; CHECK-NO: phi
; CHECK-NO-NEXT: load
  %c = phi ptr [%c3, %loop], [%a2, %if.then]
  %v = load volatile i32, ptr %c, align 4
  %iv.inc = add i32 %iv, 1
  %cmp = icmp slt i32 %iv.inc, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %v
}

; case with a loop. There is phi node.
define i32 @test8(i32 %N, i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test8
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  br label %loop

loop:
  %iv = phi i32 [0, %entry], [%iv.inc, %fallthrough]
  %c3 = phi ptr [%a1, %entry], [%c, %fallthrough]
  %b3 = phi ptr [%b1, %entry], [%b, %fallthrough]
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br label %fallthrough

fallthrough:
; CHECK: getelementptr inbounds i8, {{.+}} 40
  %c = phi ptr [%c3, %loop], [%a2, %if.then]
  %b = phi ptr [%b3, %loop], [%b2, %if.then]
  %v = load volatile i32, ptr %c, align 4
  %iv.inc = add i32 %iv, 1
  %cmp = icmp slt i32 %iv.inc, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %v
}

; case with a loop. There is phi node but it does not fit.
define i32 @test9(i32 %N, i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test9
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  br label %loop

loop:
; CHECK-LABEL: loop:
; CHECK-YES: sunk_phi
  %iv = phi i32 [0, %entry], [%iv.inc, %fallthrough]
  %c3 = phi ptr [%a1, %entry], [%c, %fallthrough]
  %b3 = phi ptr [%b1, %entry], [%b2, %fallthrough]
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br label %fallthrough

fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO-LABEL: fallthrough:
; CHECK-NO: phi
; CHECK-NO-NEXT: phi
; CHECK-NO-NEXT: load
  %c = phi ptr [%c3, %loop], [%a2, %if.then]
  %b = phi ptr [%b3, %loop], [%b2, %if.then]
  %v = load volatile i32, ptr %c, align 4
  %iv.inc = add i32 %iv, 1
  %cmp = icmp slt i32 %iv.inc, %N
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %v
}

; Case through a loop. No phi node.
define i32 @test10(i32 %N, i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test10
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br label %fallthrough

fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO-LABEL: fallthrough:
; CHECK-NO-NEXT: phi
; CHECK-NO-NEXT: br
  %c = phi ptr [%a1, %entry], [%a2, %if.then]
  br label %loop

loop:
  %iv = phi i32 [0, %fallthrough], [%iv.inc, %loop]
  %iv.inc = add i32 %iv, 1
  %cmp = icmp slt i32 %iv.inc, %N
  br i1 %cmp, label %loop, label %exit

exit:
; CHECK-YES: sunkaddr
  %v = load volatile i32, ptr %c, align 4
  ret i32 %v
}

; Case through a loop. There is a phi.
define i32 @test11(i32 %N, i1 %cond, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test11
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br label %fallthrough

fallthrough:
; CHECK: phi
; CHECK: phi
; CHECK: br
  %c = phi ptr [%a1, %entry], [%a2, %if.then]
  %b = phi ptr [%b1, %entry], [%b2, %if.then]
  br label %loop

loop:
  %iv = phi i32 [0, %fallthrough], [%iv.inc, %loop]
  %iv.inc = add i32 %iv, 1
  %cmp = icmp slt i32 %iv.inc, %N
  br i1 %cmp, label %loop, label %exit

exit:
; CHECK: sunkaddr
  %v = load volatile i32, ptr %c, align 4
  ret i32 %v
}

; Complex case with address value from previous iteration.
define i32 @test12(i32 %N, i1 %cond, ptr %b1, ptr %b2, ptr %b3) {
; CHECK-LABEL: @test12
entry:
  %a1 = getelementptr inbounds i64, ptr %b1, i64 5
  br label %loop

loop:
; CHECK-LABEL: loop:
; CHECK-YES: sunk_phi
; CHECK-NO: phi
; CHECK-NO-NEXT: phi
; CHECK-NO-NEXT: phi
; CHECK-NO-NEXT: br
  %iv = phi i32 [0, %entry], [%iv.inc, %backedge]
  %c3 = phi ptr [%a1, %entry], [%c, %backedge]
  %b4 = phi ptr [%b1, %entry], [%b5, %backedge]
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %a2 = getelementptr inbounds i64, ptr %b2, i64 5
  br label %fallthrough

fallthrough:
; CHECK-LABEL: fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO: phi
; CHECK-NO-NEXT: phi
; CHECK-NO-NEXT: load
  %c = phi ptr [%c3, %loop], [%a2, %if.then]
  %b6 = phi ptr [%b4, %loop], [%b2, %if.then]
  %v = load volatile i32, ptr %c, align 4
  %a4 = getelementptr inbounds i64, ptr %b4, i64 5
  %cmp = icmp slt i32 %iv, 20
  br i1 %cmp, label %backedge, label %if.then.2

if.then.2:
  br label %backedge

backedge:
  %b5 = phi ptr [%b4, %fallthrough], [%b6, %if.then.2]
  %iv.inc = add i32 %iv, 1
  %cmp2 = icmp slt i32 %iv.inc, %N
  br i1 %cmp2, label %loop, label %exit

exit:
  ret i32 %v
}

%struct.S = type {i32, i32}
; Case with index
define i32 @test13(i1 %cond, ptr %b1, ptr %b2, i64 %Index) {
; CHECK-LABEL: @test13
entry:
  %a1 = getelementptr inbounds %struct.S, ptr %b1, i64 %Index, i32 1
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %i2 = mul i64 %Index, 2
  %a2 = getelementptr inbounds %struct.S, ptr %b2, i64 %Index, i32 1
  br label %fallthrough

fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO-LABEL: fallthrough:
; CHECK-NO-NEXT: phi
; CHECK-NO-NEXT: load
  %a = phi ptr [%a1, %entry], [%a2, %if.then]
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

; Select of Select case.
define i64 @test14(i1 %c1, i1 %c2, ptr %b1, ptr %b2, ptr %b3) {
; CHECK-LABEL: @test14
entry:
; CHECK-LABEL: entry:
  %g1 = getelementptr inbounds i64, ptr %b1, i64 5
  %g2 = getelementptr inbounds i64, ptr %b2, i64 5
  %g3 = getelementptr inbounds i64, ptr %b3, i64 5
  %s1 = select i1 %c1, ptr %g1, ptr %g2
  %s2 = select i1 %c2, ptr %s1, ptr %g3
; CHECK: sunkaddr
  %v = load i64 , ptr %s2, align 8
  ret i64 %v
}

; Select of Phi case.
define i64 @test15(i1 %c1, i1 %c2, ptr %b1, ptr %b2, ptr %b3) {
; CHECK-LABEL: @test15
entry:
  %g1 = getelementptr inbounds i64, ptr %b1, i64 5
  %g2 = getelementptr inbounds i64, ptr %b2, i64 5
  %g3 = getelementptr inbounds i64, ptr %b3, i64 5
  br i1 %c1, label %if.then, label %fallthrough

if.then:
  br label %fallthrough

fallthrough:
; CHECK-LABEL: fallthrough:
  %p1 = phi ptr [%g1, %entry], [%g2, %if.then]
  %s1 = select i1 %c2, ptr %p1, ptr %g3
; CHECK-YES: sunkaddr
; CHECK-NO: phi
; CHECK-NO-NEXT: select
; CHECK-NO-NEXT: load
  %v = load i64 , ptr %s1, align 8
  ret i64 %v
}

; Select of Phi case. Phi exists
define i64 @test16(i1 %c1, i1 %c2, ptr %b1, ptr %b2, ptr %b3) {
; CHECK-LABEL: @test16
entry:
  %g1 = getelementptr inbounds i64, ptr %b1, i64 5
  %g2 = getelementptr inbounds i64, ptr %b2, i64 5
  %g3 = getelementptr inbounds i64, ptr %b3, i64 5
  br i1 %c1, label %if.then, label %fallthrough

if.then:
  br label %fallthrough

fallthrough:
; CHECK-LABEL: fallthrough:
  %p = phi ptr [%b1, %entry], [%b2, %if.then]
  %p1 = phi ptr [%g1, %entry], [%g2, %if.then]
  %s1 = select i1 %c2, ptr %p1, ptr %g3
; CHECK: sunkaddr
  %v = load i64 , ptr %s1, align 8
  ret i64 %v
}

; Phi of Select case.
define i64 @test17(i1 %c1, i1 %c2, ptr %b1, ptr %b2, ptr %b3) {
; CHECK-LABEL: @test17
entry:
  %g1 = getelementptr inbounds i64, ptr %b1, i64 5
  %g2 = getelementptr inbounds i64, ptr %b2, i64 5
  %g3 = getelementptr inbounds i64, ptr %b3, i64 5
  %s1 = select i1 %c2, ptr %g1, ptr %g2
  br i1 %c1, label %if.then, label %fallthrough

if.then:
  br label %fallthrough

fallthrough:
; CHECK-LABEL: fallthrough:
  %p1 = phi ptr [%s1, %entry], [%g3, %if.then]
; CHECK-YES: sunkaddr
; CHECK-NO: phi
; CHECK-NO-NEXT: load
  %v = load i64 , ptr %p1, align 8
  ret i64 %v
}

; The same two addr modes by different paths
define i32 @test18(i1 %cond1, i1 %cond2, ptr %b1, ptr %b2) {
; CHECK-LABEL: @test18
entry:
  %g1 = getelementptr inbounds i64, ptr %b2, i64 5
  br i1 %cond1, label %if.then1, label %if.then2

if.then1:
  %g2 = getelementptr inbounds i64, ptr %b1, i64 5
  br label %fallthrough

if.then2:
  br i1 %cond2, label %fallthrough, label %if.then3

if.then3:
  br label %fallthrough

fallthrough:
; CHECK-YES: sunk_phi
; CHECK-NO-LABEL: fallthrough:
; CHECK-NO: phi
; CHECK-NO-NEXT: load
  %c = phi ptr [%g2, %if.then1], [%g1, %if.then2], [%g1, %if.then3]
  %v1 = load i32, ptr %c, align 4
  %g1_1 = getelementptr inbounds i64, ptr %b2, i64 5
  %v2 = load i32, ptr %g1_1, align 4
  %v = add i32 %v1, %v2
  ret i32 %v
}

; Different types but null is the first?
define i32 @test19(i1 %cond1, i1 %cond2, i64* %b2, i8* %b1) {
; CHECK-LABEL: @test19
entry:
  %g1 = getelementptr inbounds i64, i64* %b2, i64 5
  %bc1 = bitcast i64* %g1 to i32*
  br i1 %cond1, label %if.then1, label %if.then2

if.then1:
  %g2 = getelementptr inbounds i8, i8* %b1, i64 40
  %bc2 = bitcast i8* %g2 to i32*
  br label %fallthrough

if.then2:
  %bc1_1 = bitcast i64* %g1 to i32*
  br i1 %cond2, label %fallthrough, label %if.then3

if.then3:
  %g3 = getelementptr inbounds i64, i64* null, i64 5
  %bc1_2 = bitcast i64* %g3 to i32*
  br label %fallthrough

fallthrough:
; CHECK-NOT: sunk_phi
  %c = phi i32* [%bc2, %if.then1], [%bc1_1, %if.then2], [%bc1_2, %if.then3]
  %v1 = load i32, i32* %c, align 4
  %g1_1 = getelementptr inbounds i64, i64* %b2, i64 5
  %bc1_1_1 = bitcast i64* %g1_1 to i32*
  %v2 = load i32, i32* %bc1_1_1, align 4
  %v = add i32 %v1, %v2
  ret i32 %v
}
