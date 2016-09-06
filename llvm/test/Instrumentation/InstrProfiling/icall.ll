;; Check that static counters are allocated for value profiler

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -vp-static-alloc=true -instrprof -S | FileCheck %s --check-prefix=STATIC
; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=STATIC
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -vp-static-alloc=false -instrprof -S | FileCheck %s --check-prefix=DYN
; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -vp-static-alloc=false -S | FileCheck %s --check-prefix=DYN


@__profn_foo = private constant [3 x i8] c"foo"

define i32 @foo(i32 ()* ) {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12884901887, i32 1, i32 0)
  %2 = ptrtoint i32 ()* %0 to i64
  call void @llvm.instrprof.value.profile(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12884901887, i64 %2, i32 0, i32 0)
  %3 = tail call i32 %0()
  ret i32 %3
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(i8*, i64, i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.instrprof.value.profile(i8*, i64, i64, i32, i32) #0

attributes #0 = { nounwind }

; STATIC: @__profvp_foo
; STATIC: @__llvm_prf_vnodes

; DYN-NOT: @__profvp_foo
; DYN-NOT: @__llvm_prf_vnodes
