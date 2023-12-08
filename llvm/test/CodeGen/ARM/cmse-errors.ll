; RUN: not llc -mtriple=thumbv8m.main-eabi %s -o - 2>&1 | FileCheck %s

%struct.two_ints = type { i32, i32 }
%struct.__va_list = type { ptr }

define void @test1(ptr noalias nocapture sret(%struct.two_ints) align 4 %agg.result) "cmse_nonsecure_entry" {
entry:
  store i64 8589934593, ptr %agg.result, align 4
  ret void
}
; CHECK: error: {{.*}}test1{{.*}}: secure entry function would return value through pointer

define void @test2(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) "cmse_nonsecure_entry" {
entry:
  ret void
}
; CHECK: error: {{.*}}test2{{.*}}:  secure entry function requires arguments on stack 

define void @test3(ptr nocapture %p) {
entry:
  tail call void %p(i32 1, i32 2, i32 3, i32 4, i32 5) "cmse_nonsecure_call"
  ret void
}
; CHECK: error: {{.*}}test3{{.*}}: call to non-secure function would require passing arguments on stack


define void @test4(ptr nocapture %p) {
entry:
  %r = alloca %struct.two_ints, align 4
  call void %p(ptr nonnull sret(%struct.two_ints) align 4 %r) "cmse_nonsecure_call"
  ret void
}
; CHECK: error: {{.*}}test4{{.*}}: call to non-secure function would return value through pointer

declare void @llvm.va_start(ptr) "nounwind"

declare void @llvm.va_end(ptr) "nounwind"

define i32 @test5(i32 %a, ...) "cmse_nonsecure_entry" {
entry:
  %vl = alloca %struct.__va_list, align 4
  call void @llvm.va_start(ptr nonnull %vl)
  %argp.cur = load ptr, ptr %vl, align 4
  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
  store ptr %argp.next, ptr %vl, align 4
  %0 = load i32, ptr %argp.cur, align 4
  call void @llvm.va_end(ptr nonnull %vl)
  ret i32 %0
}
; CHECK: error: {{.*}}test5{{.*}}: secure entry function must not be variadic

define void @test6(ptr nocapture %p) {
entry:
  tail call void (i32, ...) %p(i32 1, i32 2, i32 3, i32 4, i32 5) "cmse_nonsecure_call"
  ret void
}
; CHECK: error: {{.*}}test6{{.*}}: call to non-secure function would require passing arguments on stack

define void @neg_test1(ptr nocapture %p)  {
entry:
  tail call void (i32, ...) %p(i32 1, i32 2, i32 3, i32 4) "cmse_nonsecure_call"
  ret void
}

define void @neg_test2(i32 %a, ...) "cmse_nonsecure_entry" {
entry:
  ret void
}
; CHECK-NOT: error:
