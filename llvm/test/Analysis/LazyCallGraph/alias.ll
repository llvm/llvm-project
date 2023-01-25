; RUN: opt -disable-output -passes=print-lcg %s 2>&1 | FileCheck %s
;
; Aliased function should be reachable in CGSCC.

target triple = "x86_64-grtev4-linux-gnu"

; CHECK:        Edges in function: foo
; CHECK:        Edges in function: bar
; CHECK:        Edges in function: baz

; CHECK:       RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      foo
; CHECK-EMPTY:
; CHECK:       RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      bar

; CHECK-NOT:       baz

@alias1 = weak dso_local alias ptr (ptr), ptr @foo

define dso_local ptr @foo(ptr %returned) {
  ret ptr %returned
}

@alias2 = weak dso_local alias ptr (ptr), ptr @bar

define internal ptr @bar(ptr %returned) {
  ret ptr %returned
}

; Internal alias is not reachable.
@alias3 = internal alias ptr (ptr), ptr @baz

define internal ptr @baz(ptr %returned) {
  ret ptr %returned
}
