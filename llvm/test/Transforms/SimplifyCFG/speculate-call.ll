; RUN: opt -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

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

; FIXME: We should correctly drop the attribute since it may no longer be valid
; in the context the call is moved to.
define i32 @strip_attr(i32 * %p) {
; CHECK-LABEL: strip_attr  
; CHECK-LABEL: entry:
; CHECK:         %nullchk = icmp ne i32* %p, null
; CHECK:         %val = call i32 @func_deref(i32* nonnull %p)
; CHECK:         select 
entry:
  %nullchk = icmp ne i32* %p, null
  br i1 %nullchk, label %if, label %end

if:
  %val = call i32 @func_deref(i32* nonnull %p) #1
  br label %end

end:
  %ret = phi i32 [%val, %if], [0, %entry]
  ret i32 %ret
}

declare i32 @func_deref(i32*) #1

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind argmemonly speculatable }

