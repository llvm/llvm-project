; RUN: not llc -mtriple=thumbv8m.main-eabi %s -o - 2>&1 | FileCheck %s

%struct.two_ints = type { i32, i32 }
%struct.__va_list = type { i8* }

define void @test1(%struct.two_ints* noalias nocapture sret align 4 %agg.result) "cmse_nonsecure_entry" {
entry:
  %0 = bitcast %struct.two_ints* %agg.result to i64*
  store i64 8589934593, i64* %0, align 4
  ret void
}
; CHECK: error: {{.*}}test1{{.*}}: secure entry function would return value through pointer

define void @test2(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) "cmse_nonsecure_entry" {
entry:
  ret void
}
; CHECK: error: {{.*}}test2{{.*}}:  secure entry function requires arguments on stack 

define void @test3(void (i32, i32, i32, i32, i32)* nocapture %p) {
entry:
  tail call void %p(i32 1, i32 2, i32 3, i32 4, i32 5) "cmse_nonsecure_call"
  ret void
}
; CHECK: error: {{.*}}test3{{.*}}: call to non-secure function would require passing arguments on stack


define void @test4(void (%struct.two_ints*)* nocapture %p) {
entry:
  %r = alloca %struct.two_ints, align 4
  %0 = bitcast %struct.two_ints* %r to i8*
  call void %p(%struct.two_ints* nonnull sret align 4 %r) "cmse_nonsecure_call"
  ret void
}
; CHECK: error: {{.*}}test4{{.*}}: call to non-secure function would return value through pointer

declare void @llvm.va_start(i8*) "nounwind"

declare void @llvm.va_end(i8*) "nounwind"

define i32 @test5(i32 %a, ...) "cmse_nonsecure_entry" {
entry:
  %vl = alloca %struct.__va_list, align 4
  %0 = bitcast %struct.__va_list* %vl to i8*
  call void @llvm.va_start(i8* nonnull %0)
  %1 = getelementptr inbounds %struct.__va_list, %struct.__va_list* %vl, i32 0, i32 0
  %argp.cur = load i8*, i8** %1, align 4
  %argp.next = getelementptr inbounds i8, i8* %argp.cur, i32 4
  store i8* %argp.next, i8** %1, align 4
  %2 = bitcast i8* %argp.cur to i32*
  %3 = load i32, i32* %2, align 4
  call void @llvm.va_end(i8* nonnull %0)
  ret i32 %3
}
; CHECK: error: {{.*}}test5{{.*}}: secure entry function must not be variadic

define void @test6(void (i32, ...)* nocapture %p) {
entry:
  tail call void (i32, ...) %p(i32 1, i32 2, i32 3, i32 4, i32 5) "cmse_nonsecure_call"
  ret void
}
; CHECK: error: {{.*}}test6{{.*}}: call to non-secure function would require passing arguments on stack

define void @neg_test1(void (i32, ...)* nocapture %p)  {
entry:
  tail call void (i32, ...) %p(i32 1, i32 2, i32 3, i32 4) "cmse_nonsecure_call"
  ret void
}

define void @neg_test2(i32 %a, ...) "cmse_nonsecure_entry" {
entry:
  ret void
}
; CHECK-NOT: error:
