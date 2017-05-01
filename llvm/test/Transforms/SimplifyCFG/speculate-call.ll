; RUN: opt -S -simplifycfg < %s | FileCheck %s

; CHECK-LABEL: @speculatable_attribute
; CHECK: select
define i32 @speculatable_attribute(i32 %a) {
entry:
  %c = icmp sgt i32 %a, 64
  br i1 %c, label %end, label %if

if:
  %val = call i32 @func() #0
  br label %end

end:
  %ret = phi i32 [%val, %if], [0, %entry]
  ret i32 %ret
}

define i32 @func() #0 {
  ret i32 1
}
attributes #0 = { nounwind readnone speculatable }

