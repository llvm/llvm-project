; RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

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

; We should correctly drop the attribute since it may no longer be valid
; in the context the call is moved to.
; Since the function is speculatable, the nonnull attribute need not be dropped
; since it propagates poison (and call executes fine) if the parameter is indeed
; null.
define i32 @strip_attr(ptr %p) {
; CHECK-LABEL: strip_attr
; CHECK-LABEL: entry:
; CHECK:         %nullchk = icmp ne ptr %p, null
; CHECK:         %val = call i32 @func_nonnull(ptr nonnull %p)
; CHECK:         select
entry:
  %nullchk = icmp ne ptr %p, null
  br i1 %nullchk, label %if, label %end

if:
  %val = call i32 @func_nonnull(ptr nonnull %p) #1
  br label %end

end:
  %ret = phi i32 [%val, %if], [0, %entry]
  ret i32 %ret
}

; We should strip the deref attribute since it can cause UB when the
; speculatable call is moved.
define i32 @strip_attr2(ptr %p) {
; CHECK-LABEL: strip_attr2
; CHECK-LABEL: entry:
; CHECK:         %nullchk = icmp ne ptr %p, null
; CHECK:         %val = call i32 @func_nonnull(ptr %p)
; CHECK:         select
entry:
  %nullchk = icmp ne ptr %p, null
  br i1 %nullchk, label %if, label %end

if:
  %val = call i32 @func_nonnull(ptr dereferenceable(12) %p) #1
  br label %end

end:
  %ret = phi i32 [%val, %if], [0, %entry]
  ret i32 %ret
}

declare i32 @func_nonnull(ptr) #1

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind argmemonly speculatable }

