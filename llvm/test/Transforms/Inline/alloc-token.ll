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
declare [2 x i64] @alloc_coerced(i64)
declare { ptr, i8 } @find_or_insert(i64)

; A malloc_span wrapper's allocation inherits the token when the span is
; returned as a whole: directly, through a phi, or through a local temporary
; it is stored to and loaded from (as before SROA; here coerced to [2 x i64]
; as on AArch64).
define internal { ptr, i64 } @span_direct(i64 %size) alwaysinline {
  %s = call { ptr, i64 } @__size_returning_new(i64 %size)
  ret { ptr, i64 } %s
}

define internal [2 x i64] @span_temporary(i64 %size) alwaysinline {
  %retval = alloca { ptr, i64 }
  %s = call [2 x i64] @alloc_coerced(i64 %size)
  store [2 x i64] %s, ptr %retval
  %r = load [2 x i64], ptr %retval
  ret [2 x i64] %r
}

; The span from a non-allocating call, rebuilt from its fields, is not followed.
define internal { ptr, i64 } @span_phi(i64 %size, i1 %c) alwaysinline {
entry:
  br i1 %c, label %reuse, label %fresh
reuse:
  %f = call { ptr, i8 } @find_or_insert(i64 %size)
  %p = extractvalue { ptr, i8 } %f, 0
  %s0 = insertvalue { ptr, i64 } poison, ptr %p, 0
  %s1 = insertvalue { ptr, i64 } %s0, i64 %size, 1
  br label %done
fresh:
  %n = call { ptr, i64 } @__size_returning_new(i64 %size)
  br label %done
done:
  %r = phi { ptr, i64 } [ %s1, %reuse ], [ %n, %fresh ]
  ret { ptr, i64 } %r
}

; CHECK-LABEL: define void @span_whole(
; CHECK: call { ptr, i64 } @__size_returning_new(i64 4){{.*}}, !alloc_token [[MD]]
; CHECK: call [2 x i64] @alloc_coerced(i64 4){{.*}}, !alloc_token [[MD]]
; CHECK: call { ptr, i8 } @find_or_insert(i64 4){{$}}
; CHECK: call { ptr, i64 } @__size_returning_new(i64 4){{.*}}, !alloc_token [[MD]]
define void @span_whole(i1 %c) {
  %a = call { ptr, i64 } @span_direct(i64 4), !alloc_token !0
  %b = call [2 x i64] @span_temporary(i64 4), !alloc_token !0
  %d = call { ptr, i64 } @span_phi(i64 4, i1 %c), !alloc_token !0
  ret void
}

; A span assembled field by field in a temporary (as Clang emits on x86-64
; before SROA) is not followed.
define internal { ptr, i64 } @span_fields(i64 %size) alwaysinline {
  %retval = alloca { ptr, i64 }
  %s = call { ptr, i64 } @__size_returning_new(i64 %size)
  %p = extractvalue { ptr, i64 } %s, 0
  store ptr %p, ptr %retval
  %n.addr = getelementptr inbounds { ptr, i64 }, ptr %retval, i32 0, i32 1
  %n = extractvalue { ptr, i64 } %s, 1
  store i64 %n, ptr %n.addr
  %r = load { ptr, i64 }, ptr %retval
  ret { ptr, i64 } %r
}

; CHECK-LABEL: define void @span_fields_unsupported(
; CHECK: call { ptr, i64 } @__size_returning_new(i64 4){{$}}
define void @span_fields_unsupported() {
  %a = call { ptr, i64 } @span_fields(i64 4), !alloc_token !0
  ret void
}

; A pointer wrapper returning through a temporary: the call whose result is
; last stored before the load inherits the token, an earlier one does not.
define internal ptr @ptr_temporary(i64 %size) alwaysinline {
  %p = alloca ptr
  %h = call ptr @helper()
  store ptr %h, ptr %p
  %m = call ptr @malloc(i64 %size)
  store ptr %m, ptr %p
  %r = load ptr, ptr %p
  ret ptr %r
}

; A call that may write memory between the store and the load stops it.
define internal ptr @ptr_clobbered(i64 %size) alwaysinline {
  %p = alloca ptr
  %m = call ptr @malloc(i64 %size)
  store ptr %m, ptr %p
  %h = call ptr @helper()
  %r = load ptr, ptr %p
  ret ptr %r
}

; CHECK-LABEL: define void @ptr_temporaries(
; CHECK: call ptr @helper(){{$}}
; CHECK: call ptr @malloc(i64 4){{.*}}, !alloc_token [[MD]]
; CHECK: call ptr @malloc(i64 4){{$}}
; CHECK: call ptr @helper(){{$}}
define void @ptr_temporaries() {
  %a = call ptr @ptr_temporary(i64 4), !alloc_token !0
  %b = call ptr @ptr_clobbered(i64 4), !alloc_token !0
  ret void
}

; A pointer taken out of a non-span struct such as {ptr, i8} returned by a
; non-allocating call is not followed.
define internal ptr @ptr_field(i64 %size, i1 %c) alwaysinline {
  %f = call { ptr, i8 } @find_or_insert(i64 %size)
  %p = extractvalue { ptr, i8 } %f, 0
  %m = call ptr @malloc(i64 %size)
  %r = select i1 %c, ptr %p, ptr %m
  ret ptr %r
}

; CHECK-LABEL: define void @ptr_from_field(
; CHECK: call { ptr, i8 } @find_or_insert(i64 4){{$}}
; CHECK: call ptr @malloc(i64 4){{.*}}, !alloc_token [[MD]]
define void @ptr_from_field(i1 %c) {
  %a = call ptr @ptr_field(i64 4, i1 %c), !alloc_token !0
  ret void
}

; CHECK-DAG: [[MD]] = !{!"Outer", i1 true}
; CHECK-DAG: [[OWN]] = !{!"Inner", i1 false}
; CHECK-DAG: [[CALLER]] = !{!"Outer", i1 true, !"overwrite_unknown"}
!0 = !{!"Outer", i1 true}
!1 = !{!"Inner", i1 false}
!2 = !{!"Outer", i1 true, !"overwrite_unknown"}
!3 = !{!"", i1 false, !"wrapper_unknown"}
