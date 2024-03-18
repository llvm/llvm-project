;; Foo's cfg is:
;; 0->2->3
;;  ->1->3 
;; Hash is calculated by order: 0 2 1 1 2 1 3 0 3 0.

;; Profile for version 1:
; RUN: echo 'v1' > %t1
; RUN: echo 'f _Z3fooi' >> %t1
; RUN: echo 'h 1552303144703514280' >> %t1
; RUN: echo 'c 0 2' >> %t1
; RUN: echo 'c 1' >> %t1
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 \
; RUN: -mf-cfg-hash=true | FileCheck %s -check-prefix=MATCH

;; Profile for version 1:
; RUN: echo 'v1' > %t2
; RUN: echo 'f _Z3fooi' >> %t2
; RUN: echo 'h 0' >> %t2
; RUN: echo 'c 0 2' >> %t2
; RUN: echo 'c 1' >> %t2
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2 \
; RUN: -mf-cfg-hash=true | FileCheck %s -check-prefix=DISMATCH

; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2 \
; RUN: | FileCheck %s -check-prefix=MATCH

; MATCH: _Z3fooi.cold
; DISMATCH-NOT: _Z3fooi.cold

; ModuleID = 'foo.cc'
source_filename = "foo.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef i32 @_Z3fooi(i32 noundef %a) #0 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %and = and i32 %0, 1
  %tobool = icmp ne i32 %and, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %a.addr, align 4
  %call = call noundef i32 @_Z3bari(i32 noundef %1)
  store i32 %call, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  %2 = load i32, ptr %a.addr, align 4
  %call1 = call noundef i32 @_Z3bazi(i32 noundef %2)
  store i32 %call1, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %3 = load i32, ptr %retval, align 4
  ret i32 %3
}

declare noundef i32 @_Z3bari(i32 noundef) #1

declare noundef i32 @_Z3bazi(i32 noundef) #1

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 18.0.0"}