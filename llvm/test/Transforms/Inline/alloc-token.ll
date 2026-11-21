; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

declare ptr @malloc(i64)
declare ptr @helper()

define internal ptr @wrapper(i64 %size) alwaysinline {
  ; Not returned: must not inherit the token.
  %h = call ptr @helper()
  %p = call ptr @malloc(i64 %size)
  ret ptr %p
}

; CHECK-LABEL: define ptr @inherits(
; CHECK: call ptr @helper(){{$}}
; CHECK: call ptr @malloc(i64 4){{.*}}, !alloc_token [[MD:![0-9]+]]
define ptr @inherits() {
  %c = call ptr @wrapper(i64 4), !alloc_token !0
  ret ptr %c
}

define internal ptr @wrapper_own(i64 %size) alwaysinline {
  %p = call ptr @malloc(i64 %size), !alloc_token !1
  ret ptr %p
}

; Metadata the wrapper set itself is never overwritten.
; CHECK-LABEL: define ptr @no_overwrite(
; CHECK: call ptr @malloc(i64 4){{.*}}, !alloc_token [[OWN:![0-9]+]]
define ptr @no_overwrite() {
  %c = call ptr @wrapper_own(i64 4), !alloc_token !0
  ret ptr %c
}

define internal ptr @wrapper_unknown(i64 %size) alwaysinline {
  %p = call ptr @malloc(i64 %size), !alloc_token !3
  ret ptr %p
}

; Unless it denotes an unknown type (empty type name).
; CHECK-LABEL: define ptr @overwrite_unknown(
; CHECK: call ptr @malloc(i64 4){{.*}}, !alloc_token [[CALLER:![0-9]+]]
define ptr @overwrite_unknown() {
  %c = call ptr @wrapper_unknown(i64 4), !alloc_token !2
  ret ptr %c
}

declare { ptr, i64 } @__size_returning_new(i64)

; malloc_span wrappers return a {begin, size} pair. The allocation that
; produces the begin pointer inherits the token, whether it is returned
; directly, rebuilt with insertvalue, or returned through a local temporary as
; Clang emits it before SROA (e.g. when inlining an always_inline wrapper).
define internal { ptr, i64 } @span_direct(i64 %size) alwaysinline {
  %s = call { ptr, i64 } @__size_returning_new(i64 %size)
  ret { ptr, i64 } %s
}

define internal { ptr, i64 } @span_rebuilt(i64 %size) alwaysinline {
  %s = call { ptr, i64 } @__size_returning_new(i64 %size)
  %p = extractvalue { ptr, i64 } %s, 0
  %n = extractvalue { ptr, i64 } %s, 1
  %r0 = insertvalue { ptr, i64 } poison, ptr %p, 0
  %r1 = insertvalue { ptr, i64 } %r0, i64 %n, 1
  ret { ptr, i64 } %r1
}

define internal { ptr, i64 } @span_temporary(i64 %size) alwaysinline {
  %retval = alloca { ptr, i64 }
  %s = call { ptr, i64 } @__size_returning_new(i64 %size)
  %p.addr = getelementptr inbounds { ptr, i64 }, ptr %retval, i32 0, i32 0
  %p = extractvalue { ptr, i64 } %s, 0
  store ptr %p, ptr %p.addr
  %n.addr = getelementptr inbounds { ptr, i64 }, ptr %retval, i32 0, i32 1
  %n = extractvalue { ptr, i64 } %s, 1
  store i64 %n, ptr %n.addr
  %r = load { ptr, i64 }, ptr %retval
  ret { ptr, i64 } %r
}

; CHECK-LABEL: define void @span_inherits(
; CHECK: call { ptr, i64 } @__size_returning_new(i64 4){{.*}}, !alloc_token [[MD]]
; CHECK: call { ptr, i64 } @__size_returning_new(i64 4){{.*}}, !alloc_token [[MD]]
; CHECK: call { ptr, i64 } @__size_returning_new(i64 4){{.*}}, !alloc_token [[MD]]
define void @span_inherits() {
  %a = call { ptr, i64 } @span_direct(i64 4), !alloc_token !0
  %b = call { ptr, i64 } @span_rebuilt(i64 4), !alloc_token !0
  %c = call { ptr, i64 } @span_temporary(i64 4), !alloc_token !0
  ret void
}

; A span coerced by the ABI (e.g. [2 x i64] on AArch64), or a {begin, end}
; pair, is followed only as a whole: the call whose result is returned.
declare [2 x i64] @alloc_coerced(i64)
declare { ptr, ptr } @alloc_begin_end(i64)

define internal [2 x i64] @span_coerced(i64 %size) alwaysinline {
  %retval = alloca { ptr, i64 }
  %s = call [2 x i64] @alloc_coerced(i64 %size)
  store [2 x i64] %s, ptr %retval
  %r = load [2 x i64], ptr %retval
  ret [2 x i64] %r
}

define internal { ptr, ptr } @span_begin_end(i64 %size) alwaysinline {
  %s = call { ptr, ptr } @alloc_begin_end(i64 %size)
  ret { ptr, ptr } %s
}

; CHECK-LABEL: define void @span_whole(
; CHECK: call [2 x i64] @alloc_coerced(i64 4){{.*}}, !alloc_token [[MD]]
; CHECK: call { ptr, ptr } @alloc_begin_end(i64 4){{.*}}, !alloc_token [[MD]]
define void @span_whole() {
  %a = call [2 x i64] @span_coerced(i64 4), !alloc_token !0
  %b = call { ptr, ptr } @span_begin_end(i64 4), !alloc_token !0
  ret void
}

; Which field of a {pointer, pointer} pair is the begin pointer is not known,
; so a pair built from its fields inherits nothing.
define internal { ptr, ptr } @span_fields(i64 %size) alwaysinline {
  %p = call ptr @malloc(i64 %size)
  %e = call ptr @helper()
  %r0 = insertvalue { ptr, ptr } poison, ptr %p, 0
  %r1 = insertvalue { ptr, ptr } %r0, ptr %e, 1
  ret { ptr, ptr } %r1
}

; CHECK-LABEL: define void @span_fields_unknown(
; CHECK: call ptr @malloc(i64 4){{$}}
; CHECK: call ptr @helper(){{$}}
define void @span_fields_unknown() {
  %a = call { ptr, ptr } @span_fields(i64 4), !alloc_token !0
  ret void
}

; CHECK-DAG: [[MD]] = !{!"Outer", i1 true}
; CHECK-DAG: [[OWN]] = !{!"Inner", i1 false}
; CHECK-DAG: [[CALLER]] = !{!"Outer", i1 true, !"overwrite_unknown"}
!0 = !{!"Outer", i1 true}
!1 = !{!"Inner", i1 false}
!2 = !{!"Outer", i1 true, !"overwrite_unknown"}
!3 = !{!"", i1 false, !"wrapper_unknown"}
