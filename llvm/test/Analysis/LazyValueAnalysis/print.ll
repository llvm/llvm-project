; RUN: opt %s -disable-output -passes="jump-threading,print<lazy-value-info>" 2>&1 | FileCheck %s

; first to populate the values.

define i32 @constraint(i32 %a) {
; CHECK-LABEL: LVI for function 'constraint':
chklt64:
; CHECK-LABEL: chklt64:
; CHECK-NEXT: ; LatticeVal for: 'i32 %a' is: overdefined
; CHECK-NEXT: ; LatticeVal for: '  %cmp = icmp slt i32 %a, 64' in BB: '%chklt64' is: overdefined
; CHECK-NEXT: ; LatticeVal for: '  %cmp = icmp slt i32 %a, 64' in BB: '%chkgt0' is: constantrange<-1, 0>
; CHECK-NEXT: ; LatticeVal for: '  %cmp = icmp slt i32 %a, 64' in BB: '%notinbounds' is: overdefined
; CHECK-NEXT:   %cmp = icmp slt i32 %a, 64
; CHECK-NEXT: ; LatticeVal for: '  br i1 %cmp, label %chkgt0, label %notinbounds' in BB: '%chklt64' is: overdefined
; CHECK-NEXT: ; LatticeVal for: '  br i1 %cmp, label %chkgt0, label %notinbounds' in BB: '%chkgt0' is: overdefined
; CHECK-NEXT: ; LatticeVal for: '  br i1 %cmp, label %chkgt0, label %notinbounds' in BB: '%notinbounds' is: overdefined
; CHECK-NEXT:   br i1 %cmp, label %chkgt0, label %notinbounds
  %cmp = icmp slt i32 %a, 64
  br i1 %cmp, label %chkgt0, label %notinbounds

chkgt0:
; CHECK-LABEL: chkgt0:
; CHECK-NEXT: ; LatticeVal for: 'i32 %a' is: constantrange<-2147483648, 64>
; CHECK-NEXT: ; LatticeVal for: '  %cmp1 = icmp sgt i32 %a, 0' in BB: '%chkgt0' is: overdefined
; CHECK-NEXT: ; LatticeVal for: '  %cmp1 = icmp sgt i32 %a, 0' in BB: '%inbounds' is: constantrange<-1, 0>
; CHECK-NEXT:   %cmp1 = icmp sgt i32 %a, 0
; CHECK-NEXT: ; LatticeVal for: '  br i1 %cmp1, label %inbounds, label %notinbounds' in BB: '%chkgt0' is: overdefined
; CHECK-NEXT: ; LatticeVal for: '  br i1 %cmp1, label %inbounds, label %notinbounds' in BB: '%inbounds' is: overdefined
; CHECK-NEXT:   br i1 %cmp1, label %inbounds, label %notinbounds
  %cmp1 = icmp sgt i32 %a, 0
  br i1 %cmp1, label %inbounds, label %notinbounds

inbounds:
; CHECK-LABEL: inbounds:
; CHECK-NEXT: ; LatticeVal for: 'i32 %a' is: constantrange<1, 64>
; CHECK-NEXT: ; LatticeVal for: '  ret i32 %a' in BB: '%inbounds' is: overdefined
; CHECK-NEXT:   ret i32 %a
  ret i32 %a

notinbounds:
; CHECK-LABEL: notinbounds:
; CHECK-NEXT: ; LatticeVal for: 'i32 %a' is: constantrange<64, 1>
; CHECK-NEXT: ; LatticeVal for: '  %sum = add i32 %a, 64' in BB: '%notinbounds' is: constantrange<128, 65>
; CHECK-NEXT:   %sum = add i32 %a, 64
; CHECK-NEXT: ; LatticeVal for: '  ret i32 %sum' in BB: '%notinbounds' is: overdefined
; CHECK-NEXT:   ret i32 %sum
  %sum = add i32 %a, 64
  ret i32 %sum
}
