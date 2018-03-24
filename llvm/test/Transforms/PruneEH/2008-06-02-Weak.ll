; RUN: opt < %s -prune-eh -S | FileCheck %s
; RUN: opt < %s -passes='function-attrs,function(simplify-cfg)' -S | FileCheck %s

; We should not infer 'nounwind' for/from a weak function,
; since it can be overriden by throwing implementation.
;
; CHECK-LABEL: define weak void @f()
define weak void @f() {
entry:
        ret void
}

; CHECK-LABEL: define void @g()
define void @g() {
entry:
	call void @f()
	ret void
}

; CHECK-NOT: {{^}}attributes #{{[0-9].*}} nounwind
