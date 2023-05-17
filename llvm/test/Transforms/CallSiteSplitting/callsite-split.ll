; RUN: opt < %s -passes='function(callsite-splitting),cgscc(inline),function(instcombine,jump-threading)' -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linaro-linux-gnueabi"

%struct.bitmap = type { i32, ptr }

;CHECK-LABEL: @caller
;CHECK-LABEL: Top.split:
;CHECK: call void @callee(ptr null, ptr null, ptr %b_elt, i1 false)
;CHECK-LABEL: NextCond:
;CHECK: br {{.*}} label %callee.exit
;CHECK-LABEL: callee.exit:
;CHECK: call void @dummy2(ptr %a_elt)

define void @caller(i1 %c, ptr %a_elt, ptr %b_elt) {
entry:
  br label %Top

Top:
  %tobool1 = icmp eq ptr %a_elt, null
  br i1 %tobool1, label %CallSiteBB, label %NextCond

NextCond:
  %cmp = icmp ne ptr %b_elt, null
  br i1 %cmp, label %CallSiteBB, label %End

CallSiteBB:
  %p = phi i1 [0, %Top], [%c, %NextCond]
  call void @callee(ptr %a_elt, ptr %a_elt, ptr %b_elt, i1 %p)
  br label %End

End:
  ret void
}

define void @callee(ptr %dst_elt, ptr %a_elt, ptr %b_elt, i1 %c) {
entry:
  %tobool = icmp ne ptr %a_elt, null
  %tobool1 = icmp ne ptr %b_elt, null
  %or.cond = and i1 %tobool, %tobool1
  br i1 %or.cond, label %Cond, label %Big

Cond:
  %cmp = icmp eq ptr  %dst_elt, %a_elt
  br i1 %cmp, label %Small, label %Big

Small:
  call void @dummy2(ptr %a_elt)
  br label %End

Big:
  call void @dummy1(ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt)
  call void @dummy1(ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt)
  call void @dummy1(ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt)
  call void @dummy1(ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt)
  call void @dummy1(ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt)
  call void @dummy1(ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt)
  call void @dummy1(ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt, ptr %a_elt)
  br label %End

End:
  ret void
}

declare void @dummy2(ptr)
declare void @dummy1(ptr, ptr, ptr, ptr, ptr, ptr)


;CHECK-LABEL: @caller2
;CHECK-LABEL: Top.split:
;CHECK: call void @dummy4()
;CHECK-LABEL: NextCond.split:
;CHECK: call void @dummy3()
;CheCK-LABEL: CallSiteBB:
;CHECK: call void @foo(i1 %tobool1)
define void @caller2(i1 %c, ptr %a_elt, ptr %b_elt, ptr %c_elt) {
entry:
  br label %Top

Top:
  %tobool1 = icmp eq ptr %a_elt, %b_elt
  br i1 %tobool1, label %CallSiteBB, label %NextCond

NextCond:
  %cmp = icmp ne ptr %b_elt, %c_elt
  br i1 %cmp, label %CallSiteBB, label %End

CallSiteBB:
  %phi = phi i1 [0, %Top],[1, %NextCond]
  %u = call i1 @callee2(i1 %phi)
  call void @foo(i1 %u)
  br label %End

End:
  ret void
}

define i1 @callee2(i1 %b) {
entry:
  br i1 %b, label %BB1, label %BB2

BB1:
  call void @dummy3()
  br label %End

BB2:
  call void @dummy4()
  br label %End

End:
  ret i1 %b
}

declare void @dummy3()
declare void @dummy4()
declare void @foo(i1)
