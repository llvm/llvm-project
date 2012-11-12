; RUN: opt -insert-edge-profiling -o %t1 < %s
; RUN: rm -f %t1.prof_data
; RUN: lli %defaultjit -load %llvmshlibdir/libprofile_rt%shlibext %t1 \
; RUN:     -llvmprof-output %t1.prof_data
; RUN: opt -profile-file %t1.prof_data -profile-metadata-loader -S -o - < %s \
; RUN:     | FileCheck %s
; RUN: rm -f %t1.prof_data

; FIXME: profile_rt.dll could be built on win32.
; REQUIRES: loadable_module

;; func_mod - Branch taken 6 times in 7.
define i32 @func_mod(i32 %N) nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %N.addr = alloca i32, align 4
  store i32 %N, i32* %N.addr, align 4
  %0 = load i32* %N.addr, align 4
  %rem = srem i32 %0, 7
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %if.then, label %if.else
; CHECK: br i1 %tobool, label %if.then, label %if.else, !prof !0

if.then:
  store i32 1, i32* %retval
  br label %return

if.else:
  store i32 0, i32* %retval
  br label %return

return:
  %1 = load i32* %retval
  ret i32 %1
}

;; func_const_true - conditional branch which 100% taken probability.
define i32 @func_const_true(i32 %N) nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %N.addr = alloca i32, align 4
  store i32 %N, i32* %N.addr, align 4
  %0 = load i32* %N.addr, align 4
  %cmp = icmp eq i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end
; CHECK: br i1 %cmp, label %if.then, label %if.end, !prof !1

if.then:
  store i32 1, i32* %retval
  br label %return

if.end:
  store i32 0, i32* %retval
  br label %return

return:
  %1 = load i32* %retval
  ret i32 %1
}

;; func_const_true - conditional branch which 100% not-taken probability.
define i32 @func_const_false(i32 %N) nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %N.addr = alloca i32, align 4
  store i32 %N, i32* %N.addr, align 4
  %0 = load i32* %N.addr, align 4
  %cmp = icmp eq i32 %0, 1
  br i1 %cmp, label %if.then, label %if.end
; CHECK: br i1 %cmp, label %if.then, label %if.end, !prof !2

if.then:
  store i32 1, i32* %retval
  br label %return

if.end:
  store i32 0, i32* %retval
  br label %return

return:
  %1 = load i32* %retval
  ret i32 %1
}

define i32 @main(i32 %argc, i8** %argv) nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %loop = alloca i32, align 4
  store i32 0, i32* %retval
  store i32 0, i32* %loop, align 4
  br label %for.cond

for.cond:
  %0 = load i32* %loop, align 4
  %cmp = icmp slt i32 %0, 7000
  br i1 %cmp, label %for.body, label %for.end
; CHECK: br i1 %cmp, label %for.body, label %for.end, !prof !3

for.body:
  %1 = load i32* %loop, align 4
  %call = call i32 @func_mod(i32 %1)
  br label %for.inc

for.inc:
  %2 = load i32* %loop, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, i32* %loop, align 4
  br label %for.cond

for.end:
  %call1 = call i32 @func_const_true(i32 1)
  %call2 = call i32 @func_const_false(i32 0)
  ret i32 0
}

; CHECK: !0 = metadata !{metadata !"branch_weights", i32 6000, i32 1000}
; CHECK: !1 = metadata !{metadata !"branch_weights", i32 1, i32 0}
; CHECK: !2 = metadata !{metadata !"branch_weights", i32 0, i32 1}
; CHECK: !3 = metadata !{metadata !"branch_weights", i32 7000, i32 1}
; CHECK-NOT: !4
