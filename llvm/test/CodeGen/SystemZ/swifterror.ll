; RUN: llc < %s -mtriple=s390x-linux-gnu -disable-block-placement | FileCheck %s
; RUN: llc < %s -O0 -mtriple=s390x-linux-gnu -disable-block-placement | FileCheck --check-prefix=CHECK-O0 %s

declare ptr @malloc(i64)
declare void @free(ptr)
%swift_error = type {i64, i8}

; This tests the basic usage of a swifterror parameter. "foo" is the function
; that takes a swifterror parameter and "caller" is the caller of "foo".
define float @foo(ptr swifterror %error_ptr_ref) {
; CHECK-LABEL: foo:
; CHECK: lghi %r2, 16
; CHECK: brasl %r14, malloc
; CHECK: mvi 8(%r2), 1
; CHECK: lgr %r9, %r2
; CHECK-O0-LABEL: foo:
; CHECK-O0: lghi %r2, 16
; CHECK-O0: brasl %r14, malloc
; CHECK-O0: lgr [[T0:%r[0-9]+]], %r2
; CHECK-O0: mvi 8(%r2), 1
entry:
  %call = call ptr @malloc(i64 16)
  store ptr %call, ptr %error_ptr_ref
  %tmp = getelementptr inbounds i8, ptr %call, i64 8
  store i8 1, ptr %tmp
  ret float 1.0
}

; "caller" calls "foo" that takes a swifterror parameter.
define float @caller(ptr %error_ref) {
; CHECK-LABEL: caller:
; Make a copy of error_ref because r2 is getting clobbered
; CHECK-DAG: lgr %r[[REG1:[0-9]+]], %r2
; CHECK-DAG: lghi %r9, 0
; CHECK: brasl %r14, foo
; CHECK: %r2, %r9
; CHECK: jlh
; Access part of the error object and save it to error_ref
; CHECK: lb %r[[REG2:[0-9]+]], 8(%r2)
; CHECK: stc %r[[REG2]], 0(%r[[REG1]])
; CHECK: brasl %r14, free
; CHECK-O0-LABEL: caller:
; CHECK-O0: lghi %r9, 0
; CHECK-O0: brasl %r14, foo
; CHECK-O0: cghi %r9, 0
; CHECK-O0: jlh
entry:
  %error_ptr_ref = alloca swifterror ptr
  store ptr null, ptr %error_ptr_ref
  %call = call float @foo(ptr swifterror %error_ptr_ref)
  %error_from_foo = load ptr, ptr %error_ptr_ref
  %had_error_from_foo = icmp ne ptr %error_from_foo, null
  br i1 %had_error_from_foo, label %handler, label %cont
cont:
  %v1 = getelementptr inbounds %swift_error, ptr %error_from_foo, i64 0, i32 1
  %t = load i8, ptr %v1
  store i8 %t, ptr %error_ref
  br label %handler
handler:
  call void @free(ptr %error_from_foo)
  ret float 1.0
}

; "caller2" is the caller of "foo", it calls "foo" inside a loop.
define float @caller2(ptr %error_ref) {
; CHECK-LABEL: caller2:
; Make a copy of error_ref because r2 is getting clobbered
; CHECK: lgr %r[[REG1:[0-9]+]], %r2
; CHECK: lghi %r9, 0
; CHECK: brasl %r14, foo
; CHECK: cgijlh %r9, 0,
; CHECK: ceb %f0,
; CHECK: jnh
; Access part of the error object and save it to error_ref
; CHECK: lb %r[[REG2:[0-9]+]], 8(%r9)
; CHECK: stc %r[[REG2]], 0(%r[[REG1]])
; CHECK: lgr %r2, %r9
; CHECK: brasl %r14, free
; CHECK-O0-LABEL: caller2:
; CHECK-O0: lghi %r9, 0
; CHECK-O0: brasl %r14, foo
; CHECK-O0: cghi %r9, 0
; CHECK-O0: jlh
entry:
  %error_ptr_ref = alloca swifterror ptr
  br label %bb_loop
bb_loop:
  store ptr null, ptr %error_ptr_ref
  %call = call float @foo(ptr swifterror %error_ptr_ref)
  %error_from_foo = load ptr, ptr %error_ptr_ref
  %had_error_from_foo = icmp ne ptr %error_from_foo, null
  br i1 %had_error_from_foo, label %handler, label %cont
cont:
  %cmp = fcmp ogt float %call, 1.000000e+00
  br i1 %cmp, label %bb_end, label %bb_loop
bb_end:
  %v1 = getelementptr inbounds %swift_error, ptr %error_from_foo, i64 0, i32 1
  %t = load i8, ptr %v1
  store i8 %t, ptr %error_ref
  br label %handler
handler:
  call void @free(ptr %error_from_foo)
  ret float 1.0
}

; "foo_if" is a function that takes a swifterror parameter, it sets swifterror
; under a certain condition.
define float @foo_if(ptr swifterror %error_ptr_ref, i32 %cc) {
; CHECK-LABEL: foo_if:
; CHECK: cije %r2, 0
; CHECK: lghi %r2, 16
; CHECK: brasl %r14, malloc
; CHECK: mvi 8(%r2), 1
; CHECK: lgr %r9, %r2
; CHECK-NOT: %r9
; CHECK: br %r14
; CHECK-O0-LABEL: foo_if:
; spill to stack
; CHECK-O0: stg %r9, [[OFFS:[0-9]+]](%r15)
; CHECK-O0: chi %r2, 0
; CHECK-O0: je
; CHECK-O0: lghi %r2, 16
; CHECK-O0: brasl %r14, malloc
; CHECK-O0: lgr %r9, %r2
; CHECK-O0: mvi 8(%r2), 1
; CHECK-O0: br %r14
; reload from stack
; CHECK-O0: lg %r9, [[OFFS]](%r15)
; CHECK-O0: br %r14
entry:
  %cond = icmp ne i32 %cc, 0
  br i1 %cond, label %gen_error, label %normal

gen_error:
  %call = call ptr @malloc(i64 16)
  store ptr %call, ptr %error_ptr_ref
  %tmp = getelementptr inbounds i8, ptr %call, i64 8
  store i8 1, ptr %tmp
  ret float 1.0

normal:
  ret float 0.0
}

; "foo_loop" is a function that takes a swifterror parameter, it sets swifterror
; under a certain condition inside a loop.
define float @foo_loop(ptr swifterror %error_ptr_ref, i32 %cc, float %cc2) {
; CHECK-LABEL: foo_loop:
; CHECK: lr %r[[REG1:[0-9]+]], %r2
; CHECK: cije %r[[REG1]], 0
; CHECK: lghi %r2, 16
; CHECK: brasl %r14, malloc
; CHECK: mvi 8(%r2), 1
; CHECK: ceb %f8,
; CHECK: jnh
; CHECK: lgr %r9, %r2
; CHECK: br %r14
; CHECK-O0-LABEL: foo_loop:
; spill to stack
; CHECK-O0: stg %r9, [[OFFS:[0-9]+]](%r15)
; CHECK-O0: chi %r{{.*}}, 0
; CHECK-O0: je
; CHECK-O0: lghi %r2, 16
; CHECK-O0: brasl %r14, malloc
; CHECK-O0: lgr %r[[REG1:[0-9]+]], %r2
; CHECK-O0: mvi 8(%r[[REG1]]), 1
; CHECK-O0: jnh
; reload from stack
; CHECK-O0: lg %r9, [[OFFS:[0-9]+]](%r15)
; CHECK-O0: br %r14
entry:
  br label %bb_loop

bb_loop:
  %cond = icmp ne i32 %cc, 0
  br i1 %cond, label %gen_error, label %bb_cont

gen_error:
  %call = call ptr @malloc(i64 16)
  store ptr %call, ptr %error_ptr_ref
  %tmp = getelementptr inbounds i8, ptr %call, i64 8
  store i8 1, ptr %tmp
  br label %bb_cont

bb_cont:
  %cmp = fcmp ogt float %cc2, 1.000000e+00
  br i1 %cmp, label %bb_end, label %bb_loop
bb_end:
  ret float 0.0
}

%struct.S = type { i32, i32, i32, i32, i32, i32 }

; "foo_sret" is a function that takes a swifterror parameter, it also has a sret
; parameter.
define void @foo_sret(ptr sret(%struct.S) %agg.result, i32 %val1, ptr swifterror %error_ptr_ref) {
; CHECK-LABEL: foo_sret:
; CHECK-DAG: lgr %r[[REG1:[0-9]+]], %r2
; CHECK-DAG: lr %r[[REG2:[0-9]+]], %r3
; CHECK-DAG: lghi %r2, 16
; CHECK: brasl %r14, malloc
; CHECK: mvi 8(%r2), 1
; CHECK: st %r[[REG2]], 4(%r[[REG1]])
; CHECK: lgr %r9, %r2
; CHECK-NOT: %r9
; CHECK: br %r14

; CHECK-O0-LABEL: foo_sret:
; spill sret to stack
; CHECK-O0-DAG: stg %r2, [[OFFS1:[0-9]+]](%r15)
; CHECK-O0-DAG: st %r3, [[OFFS2:[0-9]+]](%r15)
; CHECK-O0: lghi %r2, 16
; CHECK-O0: brasl %r14, malloc
; CHECK-O0-DAG: lgr %r[[REG3:[0-9]+]], %r2
; CHECK-O0-DAG: mvi 8(%r[[REG3]]), 1
; CHECK-O0-DAG: lg %r[[REG1:[0-9]+]], [[OFFS1]](%r15)
; CHECK-O0-DAG: lgr %r9, %r[[REG3]]
; CHECK-O0-DAG: l %r[[REG2:[0-9]+]], [[OFFS2]](%r15)
; CHECK-O0: st %r[[REG2]], 4(%r[[REG1]])
; CHECK-O0: br %r14
entry:
  %call = call ptr @malloc(i64 16)
  store ptr %call, ptr %error_ptr_ref
  %tmp = getelementptr inbounds i8, ptr %call, i64 8
  store i8 1, ptr %tmp
  %v2 = getelementptr inbounds %struct.S, ptr %agg.result, i32 0, i32 1
  store i32 %val1, ptr %v2
  ret void
}

; "caller3" calls "foo_sret" that takes a swifterror parameter.
define float @caller3(ptr %error_ref) {
; CHECK-LABEL: caller3:
; Make a copy of error_ref because r2 is getting clobbered
; CHECK: lgr %r[[REG1:[0-9]+]], %r2
; CHECK: lhi %r3, 1
; CHECK: lghi %r9, 0
; CHECK: brasl %r14, foo_sret
; CHECK: jlh
; Access part of the error object and save it to error_ref
; CHECK: lb %r0, 8(%r2)
; CHECK: stc %r0, 0(%r[[REG1]])
; CHECK: brasl %r14, free

; CHECK-O0-LABEL: caller3:
; CHECK-O0: lghi %r9, 0
; CHECK-O0: lhi %r3, 1
; CHECK-O0: brasl %r14, foo_sret
; CHECK-O0: lgr {{.*}}, %r9
; CHECK-O0: cghi %r9, 0
; CHECK-O0: jlh
; Access part of the error object and save it to error_ref
; CHECK-O0: lb %r0, 8(%r{{.*}})
; CHECK-O0: stc %r0, 0(%r{{.*}})
; reload from stack
; CHECK-O0: lg %r2, {{.*}}(%r15)
; CHECK-O0: brasl %r14, free
entry:
  %s = alloca %struct.S, align 8
  %error_ptr_ref = alloca swifterror ptr
  store ptr null, ptr %error_ptr_ref
  call void @foo_sret(ptr sret(%struct.S) %s, i32 1, ptr swifterror %error_ptr_ref)
  %error_from_foo = load ptr, ptr %error_ptr_ref
  %had_error_from_foo = icmp ne ptr %error_from_foo, null
  br i1 %had_error_from_foo, label %handler, label %cont
cont:
  %v1 = getelementptr inbounds %swift_error, ptr %error_from_foo, i64 0, i32 1
  %t = load i8, ptr %v1
  store i8 %t, ptr %error_ref
  br label %handler
handler:
  call void @free(ptr %error_from_foo)
  ret float 1.0
}

; This is a caller with multiple swifterror values, it calls "foo" twice, each
; time with a different swifterror value, from "alloca swifterror".
define float @caller_with_multiple_swifterror_values(ptr %error_ref, ptr %error_ref2) {
; CHECK-LABEL: caller_with_multiple_swifterror_values:
; CHECK-DAG: lgr %r[[REG1:[0-9]+]], %r2
; CHECK-DAG: lgr %r[[REG2:[0-9]+]], %r3
; The first swifterror value:
; CHECK-DAG: lghi %r9, 0
; CHECK: brasl %r14, foo
; CHECK: ltgr %r2, %r9
; CHECK: jlh
; Access part of the error object and save it to error_ref
; CHECK: lb %r0, 8(%r2)
; CHECK: stc %r0, 0(%r[[REG1]])
; CHECK: brasl %r14, free

; The second swifterror value:
; CHECK: lghi %r9, 0
; CHECK: brasl %r14, foo
; CHECK: ltgr %r2, %r9
; CHECK: jlh
; Access part of the error object and save it to error_ref
; CHECK: lb %r0, 8(%r2)
; CHECK: stc %r0, 0(%r[[REG2]])
; CHECK: brasl %r14, free

; CHECK-O0-LABEL: caller_with_multiple_swifterror_values:

; The first swifterror value:
; CHECK-O0: lghi %r9, 0
; CHECK-O0: brasl %r14, foo
; CHECK-O0: jlh

; The second swifterror value:
; CHECK-O0: lghi %r9, 0
; CHECK-O0: brasl %r14, foo
; CHECK-O0: jlh
entry:
  %error_ptr_ref = alloca swifterror ptr
  store ptr null, ptr %error_ptr_ref
  %call = call float @foo(ptr swifterror %error_ptr_ref)
  %error_from_foo = load ptr, ptr %error_ptr_ref
  %had_error_from_foo = icmp ne ptr %error_from_foo, null
  br i1 %had_error_from_foo, label %handler, label %cont
cont:
  %v1 = getelementptr inbounds %swift_error, ptr %error_from_foo, i64 0, i32 1
  %t = load i8, ptr %v1
  store i8 %t, ptr %error_ref
  br label %handler
handler:
  call void @free(ptr %error_from_foo)

  %error_ptr_ref2 = alloca swifterror ptr
  store ptr null, ptr %error_ptr_ref2
  %call2 = call float @foo(ptr swifterror %error_ptr_ref2)
  %error_from_foo2 = load ptr, ptr %error_ptr_ref2
  %had_error_from_foo2 = icmp ne ptr %error_from_foo2, null
  br i1 %had_error_from_foo2, label %handler2, label %cont2
cont2:
  %v2 = getelementptr inbounds %swift_error, ptr %error_from_foo2, i64 0, i32 1
  %t2 = load i8, ptr %v2
  store i8 %t2, ptr %error_ref2
  br label %handler2
handler2:
  call void @free(ptr %error_from_foo2)

  ret float 1.0
}
