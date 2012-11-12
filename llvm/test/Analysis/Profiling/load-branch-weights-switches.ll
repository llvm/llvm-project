; RUN: opt -insert-edge-profiling -o %t1 < %s
; RUN: rm -f %t1.prof_data
; RUN: lli %defaultjit -load %llvmshlibdir/libprofile_rt%shlibext %t1 \
; RUN:     -llvmprof-output %t1.prof_data
; RUN: opt -profile-file %t1.prof_data -profile-metadata-loader -S -o - < %s \
; RUN:     | FileCheck %s
; RUN: rm -f %t1.prof_data

; FIXME: profile_rt.dll could be built on win32.
; REQUIRES: loadable_module

;; func_switch - Test branch probabilities for a switch instruction with an
;; even chance of taking each case (or no case).
define i32 @func_switch(i32 %N) nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %N.addr = alloca i32, align 4
  store i32 %N, i32* %N.addr, align 4
  %0 = load i32* %N.addr, align 4
  %rem = srem i32 %0, 4
  switch i32 %rem, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ]
; CHECK: ], !prof !0

sw.bb:
  store i32 5, i32* %retval
  br label %return

sw.bb1:
  store i32 6, i32* %retval
  br label %return

sw.bb2:
  store i32 7, i32* %retval
  br label %return

sw.epilog:
  store i32 8, i32* %retval
  br label %return

return:
  %1 = load i32* %retval
  ret i32 %1
}

;; func_switch_switch - Test branch probabilities in a switch-instruction that
;; leads to further switch instructions.  The first-tier switch occludes some
;; possibilities in the second-tier switches, leading to some branches having a
;; 0 probability.
define i32 @func_switch_switch(i32 %N) nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %N.addr = alloca i32, align 4
  store i32 %N, i32* %N.addr, align 4
  %0 = load i32* %N.addr, align 4
  %rem = srem i32 %0, 2
  switch i32 %rem, label %sw.default11 [
    i32 0, label %sw.bb
    i32 1, label %sw.bb5
  ]
; CHECK: ], !prof !1

sw.bb:
  %1 = load i32* %N.addr, align 4
  %rem1 = srem i32 %1, 4
  switch i32 %rem1, label %sw.default [
    i32 0, label %sw.bb2
    i32 1, label %sw.bb3
    i32 2, label %sw.bb4
  ]
; CHECK: ], !prof !2

sw.bb2:
  store i32 5, i32* %retval
  br label %return

sw.bb3:
  store i32 6, i32* %retval
  br label %return

sw.bb4:
  store i32 7, i32* %retval
  br label %return

sw.default:
  store i32 8, i32* %retval
  br label %return

sw.bb5:
  %2 = load i32* %N.addr, align 4
  %rem6 = srem i32 %2, 4
  switch i32 %rem6, label %sw.default10 [
    i32 0, label %sw.bb7
    i32 1, label %sw.bb8
    i32 2, label %sw.bb9
  ]
; CHECK: ], !prof !3

sw.bb7:
  store i32 9, i32* %retval
  br label %return

sw.bb8:
  store i32 10, i32* %retval
  br label %return

sw.bb9:
  store i32 11, i32* %retval
  br label %return

sw.default10:
  store i32 12, i32* %retval
  br label %return

sw.default11:
  store i32 13, i32* %retval
  br label %return

return:
  %3 = load i32* %retval
  ret i32 %3
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
  %cmp = icmp slt i32 %0, 4000
  br i1 %cmp, label %for.body, label %for.end
; CHECK: br i1 %cmp, label %for.body, label %for.end, !prof !4

for.body:
  %1 = load i32* %loop, align 4
  %call = call i32 @func_switch(i32 %1)
  %2 = load i32* %loop, align 4
  %call1 = call i32 @func_switch_switch(i32 %2)
  br label %for.inc

for.inc:
  %3 = load i32* %loop, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, i32* %loop, align 4
  br label %for.cond

for.end:
  ret i32 0
}

; CHECK: !0 = metadata !{metadata !"branch_weights", i32 1000, i32 1000, i32 1000, i32 1000}
; CHECK: !1 = metadata !{metadata !"branch_weights", i32 0, i32 2000, i32 2000}
; CHECK: !2 = metadata !{metadata !"branch_weights", i32 0, i32 1000, i32 0, i32 1000}
; CHECK: !3 = metadata !{metadata !"branch_weights", i32 1000, i32 0, i32 1000, i32 0}
; CHECK: !4 = metadata !{metadata !"branch_weights", i32 4000, i32 1}
; CHECK-NOT: !5
