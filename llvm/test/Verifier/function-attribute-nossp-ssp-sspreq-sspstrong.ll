; RUN: not opt -verify -o - %s 2>&1 | FileCheck %s
define void @test_1 () #1 { ret void }
define void @test_2 () #2 { ret void }
define void @test_3 () #3 { ret void }
define void @test_4 () #4 { ret void }
define void @test_5 () #5 { ret void }
define void @test_6 () #6 { ret void }
define void @test_7 () #7 { ret void }
define void @test_8 () #8 { ret void }
define void @test_9 () #9 { ret void }
define void @test_10 () #10 { ret void }
define void @test_11 () #10 { ret void }
define void @test_12 () #10 { ret void }
define void @test_13 () #10 { ret void }
define void @test_14 () #10 { ret void }

attributes #0 = { nossp }
attributes #1 = { ssp }
attributes #2 = { sspreq }
attributes #3 = { sspstrong }

attributes #4 = { nossp ssp }
attributes #5 = { nossp sspreq }
attributes #6 = { nossp sspstrong }

attributes #7 = { ssp sspreq }
attributes #8 = { ssp sspstrong }

attributes #9 = { sspreq sspstrong }

attributes #10 = { nossp ssp sspreq }
attributes #11 = { nossp ssp sspstrong }
attributes #12 = { nossp sspreq sspstrong }
attributes #13 = { ssp sspreq sspstrong }
attributes #14 = { nossp ssp sspreq sspstrong }

; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_4
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_5
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_6
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_7
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_8
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_9
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_10
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_11
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_12
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_13
; CHECK: fn attrs are mutually exclusive
; CHECK-NEXT: void ()* @test_14
