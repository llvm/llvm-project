; RUN: opt -S < %s -simplifycfg -simplifycfg-merge-cond-stores=true -simplifycfg-merge-cond-stores-aggressively=false -phi-node-folding-threshold=2 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabihf"

; This is a bit reversal that has been run through the early optimizer (-mem2reg -gvn -instcombine).
; There should be no additional PHIs created at all. The store should be on its own in a predicated
; block and there should be no PHIs.

; CHECK-LABEL: @f
; Exactly 15 phis, as there are 15 in the original test case.
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK: select
; CHECK-NOT: select
; CHECK: br i1 {{.*}}, label %[[L:.*]], label %[[R:.*]]
; CHECK: [[L]]: ; preds =
; CHECK-NEXT: store
; CHECK-NEXT: br label %[[R]]
; CHECK: [[R]]: ; preds =
; CHECK-NEXT: ret i32 0

define i32 @f(i32* %b) {
entry:
  %0 = load i32, i32* %b, align 4
  %and = and i32 %0, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %or = or i32 %0, -2147483648
  store i32 %or, i32* %b, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %1 = phi i32 [ %0, %entry ], [ %or, %if.then ]
  %and1 = and i32 %1, 2
  %tobool2 = icmp eq i32 %and1, 0
  br i1 %tobool2, label %if.end5, label %if.then3

if.then3:                                         ; preds = %if.end
  %or4 = or i32 %1, 1073741824
  store i32 %or4, i32* %b, align 4
  br label %if.end5

if.end5:                                          ; preds = %if.end, %if.then3
  %2 = phi i32 [ %1, %if.end ], [ %or4, %if.then3 ]
  %and6 = and i32 %2, 4
  %tobool7 = icmp eq i32 %and6, 0
  br i1 %tobool7, label %if.end10, label %if.then8

if.then8:                                         ; preds = %if.end5
  %or9 = or i32 %2, 536870912
  store i32 %or9, i32* %b, align 4
  br label %if.end10

if.end10:                                         ; preds = %if.end5, %if.then8
  %3 = phi i32 [ %2, %if.end5 ], [ %or9, %if.then8 ]
  %and11 = and i32 %3, 8
  %tobool12 = icmp eq i32 %and11, 0
  br i1 %tobool12, label %if.end15, label %if.then13

if.then13:                                        ; preds = %if.end10
  %or14 = or i32 %3, 268435456
  store i32 %or14, i32* %b, align 4
  br label %if.end15

if.end15:                                         ; preds = %if.end10, %if.then13
  %4 = phi i32 [ %3, %if.end10 ], [ %or14, %if.then13 ]
  %and16 = and i32 %4, 16
  %tobool17 = icmp eq i32 %and16, 0
  br i1 %tobool17, label %if.end20, label %if.then18

if.then18:                                        ; preds = %if.end15
  %or19 = or i32 %4, 134217728
  store i32 %or19, i32* %b, align 4
  br label %if.end20

if.end20:                                         ; preds = %if.end15, %if.then18
  %5 = phi i32 [ %4, %if.end15 ], [ %or19, %if.then18 ]
  %and21 = and i32 %5, 32
  %tobool22 = icmp eq i32 %and21, 0
  br i1 %tobool22, label %if.end25, label %if.then23

if.then23:                                        ; preds = %if.end20
  %or24 = or i32 %5, 67108864
  store i32 %or24, i32* %b, align 4
  br label %if.end25

if.end25:                                         ; preds = %if.end20, %if.then23
  %6 = phi i32 [ %5, %if.end20 ], [ %or24, %if.then23 ]
  %and26 = and i32 %6, 64
  %tobool27 = icmp eq i32 %and26, 0
  br i1 %tobool27, label %if.end30, label %if.then28

if.then28:                                        ; preds = %if.end25
  %or29 = or i32 %6, 33554432
  store i32 %or29, i32* %b, align 4
  br label %if.end30

if.end30:                                         ; preds = %if.end25, %if.then28
  %7 = phi i32 [ %6, %if.end25 ], [ %or29, %if.then28 ]
  %and31 = and i32 %7, 256
  %tobool32 = icmp eq i32 %and31, 0
  br i1 %tobool32, label %if.end35, label %if.then33

if.then33:                                        ; preds = %if.end30
  %or34 = or i32 %7, 8388608
  store i32 %or34, i32* %b, align 4
  br label %if.end35

if.end35:                                         ; preds = %if.end30, %if.then33
  %8 = phi i32 [ %7, %if.end30 ], [ %or34, %if.then33 ]
  %and36 = and i32 %8, 512
  %tobool37 = icmp eq i32 %and36, 0
  br i1 %tobool37, label %if.end40, label %if.then38

if.then38:                                        ; preds = %if.end35
  %or39 = or i32 %8, 4194304
  store i32 %or39, i32* %b, align 4
  br label %if.end40

if.end40:                                         ; preds = %if.end35, %if.then38
  %9 = phi i32 [ %8, %if.end35 ], [ %or39, %if.then38 ]
  %and41 = and i32 %9, 1024
  %tobool42 = icmp eq i32 %and41, 0
  br i1 %tobool42, label %if.end45, label %if.then43

if.then43:                                        ; preds = %if.end40
  %or44 = or i32 %9, 2097152
  store i32 %or44, i32* %b, align 4
  br label %if.end45

if.end45:                                         ; preds = %if.end40, %if.then43
  %10 = phi i32 [ %9, %if.end40 ], [ %or44, %if.then43 ]
  %and46 = and i32 %10, 2048
  %tobool47 = icmp eq i32 %and46, 0
  br i1 %tobool47, label %if.end50, label %if.then48

if.then48:                                        ; preds = %if.end45
  %or49 = or i32 %10, 1048576
  store i32 %or49, i32* %b, align 4
  br label %if.end50

if.end50:                                         ; preds = %if.end45, %if.then48
  %11 = phi i32 [ %10, %if.end45 ], [ %or49, %if.then48 ]
  %and51 = and i32 %11, 4096
  %tobool52 = icmp eq i32 %and51, 0
  br i1 %tobool52, label %if.end55, label %if.then53

if.then53:                                        ; preds = %if.end50
  %or54 = or i32 %11, 524288
  store i32 %or54, i32* %b, align 4
  br label %if.end55

if.end55:                                         ; preds = %if.end50, %if.then53
  %12 = phi i32 [ %11, %if.end50 ], [ %or54, %if.then53 ]
  %and56 = and i32 %12, 8192
  %tobool57 = icmp eq i32 %and56, 0
  br i1 %tobool57, label %if.end60, label %if.then58

if.then58:                                        ; preds = %if.end55
  %or59 = or i32 %12, 262144
  store i32 %or59, i32* %b, align 4
  br label %if.end60

if.end60:                                         ; preds = %if.end55, %if.then58
  %13 = phi i32 [ %12, %if.end55 ], [ %or59, %if.then58 ]
  %and61 = and i32 %13, 16384
  %tobool62 = icmp eq i32 %and61, 0
  br i1 %tobool62, label %if.end65, label %if.then63

if.then63:                                        ; preds = %if.end60
  %or64 = or i32 %13, 131072
  store i32 %or64, i32* %b, align 4
  br label %if.end65

if.end65:                                         ; preds = %if.end60, %if.then63
  %14 = phi i32 [ %13, %if.end60 ], [ %or64, %if.then63 ]
  %and66 = and i32 %14, 32768
  %tobool67 = icmp eq i32 %and66, 0
  br i1 %tobool67, label %if.end70, label %if.then68

if.then68:                                        ; preds = %if.end65
  %or69 = or i32 %14, 65536
  store i32 %or69, i32* %b, align 4
  br label %if.end70

if.end70:                                         ; preds = %if.end65, %if.then68
  %15 = phi i32 [ %14, %if.end65 ], [ %or69, %if.then68 ]
  %and71 = and i32 %15, 128
  %tobool72 = icmp eq i32 %and71, 0
  br i1 %tobool72, label %if.end75, label %if.then73

if.then73:                                        ; preds = %if.end70
  %or74 = or i32 %15, 16777216
  store i32 %or74, i32* %b, align 4
  br label %if.end75

if.end75:                                         ; preds = %if.end70, %if.then73
  ret i32 0
}
