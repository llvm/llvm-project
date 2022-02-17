; RUN: opt -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -switch-range-to-icmp < %s | FileCheck %s

define zeroext i1 @test1(i32 %x) nounwind readnone ssp noredzone {
entry:
 switch i32 %x, label %lor.rhs [
   i32 2, label %lor.end
   i32 1, label %lor.end
   i32 3, label %lor.end
 ]

lor.rhs:
 br label %lor.end

lor.end:
 %0 = phi i1 [ true, %entry ], [ false, %lor.rhs ], [ true, %entry ], [ true, %entry ]
 ret i1 %0

; CHECK-LABEL: @test1(
; CHECK: %x.off = add i32 %x, -1
; CHECK: %switch = icmp ult i32 %x.off, 3
}

define zeroext i1 @test2(i32 %x) nounwind readnone ssp noredzone {
entry:
 switch i32 %x, label %lor.rhs [
   i32 0, label %lor.end
   i32 1, label %lor.end
 ]

lor.rhs:
 br label %lor.end

lor.end:
 %0 = phi i1 [ true, %entry ], [ false, %lor.rhs ], [ true, %entry ]
 ret i1 %0

; CHECK-LABEL: @test2(
; CHECK: %switch = icmp ult i32 %x, 2
}

define i32 @test3(i1 %flag) {
entry:
 switch i1 %flag, label %bad [
   i1 true, label %good
   i1 false, label %good
 ]

good:
 ret i32 0

bad:
 ret i32 1

; CHECK-LABEL: @test3(
; CHECK: entry:
; CHECK-NEXT: ret i32 0
}
