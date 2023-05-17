; RUN: opt -S -passes=reassociate < %s | FileCheck %s
; rdar://9167457

; Reassociate shouldn't break this testcase involving a secondary
; reassociation.

; CHECK:     define
; CHECK-NOT: undef
; CHECK:     %factor = mul i32 %tmp3, -2
; CHECK-NOT: undef
; CHECK:     }

define void @x0f2f640ab6718391b59ce96d9fdeda54(i32 %arg, i32 %arg1, i32 %arg2, ptr %.out) nounwind {
_:
  %tmp = sub i32 %arg, %arg1
  %tmp3 = mul i32 %tmp, -1268345047
  %tmp4 = add i32 %tmp3, 2014710503
  %tmp5 = add i32 %tmp3, -1048397418
  %tmp6 = sub i32 %tmp4, %tmp5
  %tmp7 = sub i32 -2014710503, %tmp3
  %tmp8 = add i32 %tmp6, %tmp7
  store i32 %tmp8, ptr %.out
  ret void
}
