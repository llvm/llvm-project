; Test to ensure that the inlining cost model isn't tripped up by dead constant users.
; In this case, the call to g via a bitcasted function pointer is canonicalized to
; a direct call to g with bitcasted arguments, leaving the original bitcast
; as a dead use of g.

; RUN: opt < %s  -passes='function(instcombine),cgscc(inline)' -pass-remarks=inline -S 2>&1 \
; RUN:     | FileCheck %s

; Inline costs of f and g should be the same.

; CHECK: 'f' inlined into 'h' with (cost=[[EXPECTED_COST:.+]], threshold={{.+}})
; CHECK: 'g' inlined into 'h' with (cost=[[EXPECTED_COST]], threshold={{.+}})

%0 = type { i64, i64, i64 }
%1 = type { i64, i64, i64 }

define internal void @f(ptr align 8 %a) unnamed_addr {
start:
  ret void
}

define internal void @g(ptr align 8 %a) unnamed_addr {
start:
  ret void
}

define void @h(ptr align 8 %a, ptr align 8 %b) unnamed_addr {
start:
  call void @f(ptr align 8 %a)
  call void @g(ptr align 8 %b)
  ret void
}
