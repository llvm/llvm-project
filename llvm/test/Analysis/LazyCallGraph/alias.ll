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

@alias1 = weak dso_local alias i8* (i8*), i8* (i8*)* @foo

define dso_local i8* @foo(i8* %returned) {
  ret i8* %returned
}

@alias2 = weak dso_local alias i8* (i8*), i8* (i8*)* @bar

define internal i8* @bar(i8* %returned) {
  ret i8* %returned
}

; Internal alias is not reachable.
@alias3 = internal alias i8* (i8*), i8* (i8*)* @baz

define internal i8* @baz(i8* %returned) {
  ret i8* %returned
}
