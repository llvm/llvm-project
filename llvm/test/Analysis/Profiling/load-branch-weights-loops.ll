; RUN: opt -insert-edge-profiling -o %t1 < %s
; RUN: rm -f %t1.prof_data
; RUN: lli %defaultjit -load %llvmshlibdir/libprofile_rt%shlibext %t1 \
; RUN:     -llvmprof-output %t1.prof_data
; RUN: opt -profile-file %t1.prof_data -profile-metadata-loader -S -o - < %s \
; RUN:     | FileCheck %s
; RUN: rm -f %t1.prof_data

; FIXME: profile_rt.dll could be built on win32.
; REQUIRES: loadable_module

;; func_for - Test branch probabilities for a vanilla for loop.
define i32 @func_for(i32 %N) nounwind uwtable {
entry:
  %N.addr = alloca i32, align 4
  %ret = alloca i32, align 4
  %loop = alloca i32, align 4
  store i32 %N, i32* %N.addr, align 4
  store i32 0, i32* %ret, align 4
  store i32 0, i32* %loop, align 4
  br label %for.cond

for.cond:
  %0 = load i32* %loop, align 4
  %1 = load i32* %N.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end
; CHECK: br i1 %cmp, label %for.body, label %for.end, !prof !0

for.body:
  %2 = load i32* %N.addr, align 4
  %3 = load i32* %ret, align 4
  %add = add nsw i32 %3, %2
  store i32 %add, i32* %ret, align 4
  br label %for.inc

for.inc:
  %4 = load i32* %loop, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, i32* %loop, align 4
  br label %for.cond

for.end:
  %5 = load i32* %ret, align 4
  ret i32 %5
}

;; func_for_odd - Test branch probabilities for a for loop with a continue and
;; a break.
define i32 @func_for_odd(i32 %N) nounwind uwtable {
entry:
  %N.addr = alloca i32, align 4
  %ret = alloca i32, align 4
  %loop = alloca i32, align 4
  store i32 %N, i32* %N.addr, align 4
  store i32 0, i32* %ret, align 4
  store i32 0, i32* %loop, align 4
  br label %for.cond

for.cond:
  %0 = load i32* %loop, align 4
  %1 = load i32* %N.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end
; CHECK: br i1 %cmp, label %for.body, label %for.end, !prof !1

for.body:
  %2 = load i32* %loop, align 4
  %rem = srem i32 %2, 10
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %if.then, label %if.end
; CHECK: br i1 %tobool, label %if.then, label %if.end, !prof !2

if.then:
  br label %for.inc

if.end:
  %3 = load i32* %loop, align 4
  %cmp1 = icmp eq i32 %3, 500
  br i1 %cmp1, label %if.then2, label %if.end3
; CHECK: br i1 %cmp1, label %if.then2, label %if.end3, !prof !3

if.then2:
  br label %for.end

if.end3:
  %4 = load i32* %N.addr, align 4
  %5 = load i32* %ret, align 4
  %add = add nsw i32 %5, %4
  store i32 %add, i32* %ret, align 4
  br label %for.inc

for.inc:
  %6 = load i32* %loop, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, i32* %loop, align 4
  br label %for.cond

for.end:
  %7 = load i32* %ret, align 4
  ret i32 %7
}

;; func_while - Test branch probability in a vanilla while loop.
define i32 @func_while(i32 %N) nounwind uwtable {
entry:
  %N.addr = alloca i32, align 4
  %ret = alloca i32, align 4
  %loop = alloca i32, align 4
  store i32 %N, i32* %N.addr, align 4
  store i32 0, i32* %ret, align 4
  store i32 0, i32* %loop, align 4
  br label %while.cond

while.cond:
  %0 = load i32* %loop, align 4
  %1 = load i32* %N.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %while.body, label %while.end
; CHECK: br i1 %cmp, label %while.body, label %while.end, !prof !0

while.body:
  %2 = load i32* %N.addr, align 4
  %3 = load i32* %ret, align 4
  %add = add nsw i32 %3, %2
  store i32 %add, i32* %ret, align 4
  %4 = load i32* %loop, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, i32* %loop, align 4
  br label %while.cond

while.end:
  %5 = load i32* %ret, align 4
  ret i32 %5
}

;; func_while - Test branch probability in a vanilla do-while loop.
define i32 @func_do_while(i32 %N) nounwind uwtable {
entry:
  %N.addr = alloca i32, align 4
  %ret = alloca i32, align 4
  %loop = alloca i32, align 4
  store i32 %N, i32* %N.addr, align 4
  store i32 0, i32* %ret, align 4
  store i32 0, i32* %loop, align 4
  br label %do.body

do.body:
  %0 = load i32* %N.addr, align 4
  %1 = load i32* %ret, align 4
  %add = add nsw i32 %1, %0
  store i32 %add, i32* %ret, align 4
  %2 = load i32* %loop, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, i32* %loop, align 4
  br label %do.cond

do.cond:
  %3 = load i32* %loop, align 4
  %4 = load i32* %N.addr, align 4
  %cmp = icmp slt i32 %3, %4
  br i1 %cmp, label %do.body, label %do.end
; CHECK: br i1 %cmp, label %do.body, label %do.end, !prof !4

do.end:
  %5 = load i32* %ret, align 4
  ret i32 %5
}

define i32 @main(i32 %argc, i8** %argv) nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval
  %call = call i32 @func_for(i32 1000)
  %call1 = call i32 @func_for_odd(i32 1000)
  %call2 = call i32 @func_while(i32 1000)
  %call3 = call i32 @func_do_while(i32 1000)
  ret i32 0
}

!0 = metadata !{metadata !"branch_weights", i32 1000, i32 1}
!1 = metadata !{metadata !"branch_weights", i32 501, i32 0}
!2 = metadata !{metadata !"branch_weights", i32 450, i32 51}
!3 = metadata !{metadata !"branch_weights", i32 1, i32 50}
!4 = metadata !{metadata !"branch_weights", i32 999, i32 1}
; CHECK-NOT: !5
