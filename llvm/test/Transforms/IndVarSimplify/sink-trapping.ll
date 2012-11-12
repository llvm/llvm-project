; RUN: opt < %s -indvars -S | FileCheck %s --check-prefix=CHECK

declare i1 @b()

define i32 @a(i32 %x) nounwind {
for.body.preheader:
    %y = sdiv i32 10, %x
	br label %for.body

for.body:
    %cmp = call i1 @b()
	br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:
	ret i32 %y
}
; CHECK: for.end.loopexit:
; CHECK: sdiv
; CHECK: ret
