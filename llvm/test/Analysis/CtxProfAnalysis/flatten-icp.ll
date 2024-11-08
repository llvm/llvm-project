; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromJSON --input %t/profile.json --output %t/profile.ctxprofdata
;
; In the given profile, in one of the contexts the indirect call is taken, the
; target we're trying to ICP - GUID:2000 - doesn't appear at all. That should
; contribute to the count of the "indirect call BB".
; RUN: opt %t/test.ll -S -passes='require<ctx-prof-analysis>,module-inline,ctx-prof-flatten' -use-ctx-profile=%t/profile.ctxprofdata -ctx-prof-promote-alwaysinline 

; CHECK-LABEL: define i32 @caller(ptr %c)
; CHECK-NEXT:     [[CND:[0-9]+]] = icmp eq ptr %c, @one
; CHECK-NEXT:     br i1 [[CND]], label %{{.*}}, label %{{.*}}, !prof ![[BW:[0-9]+]]

; CHECK: ![[BW]] = !{!"branch_weights", i32 10, i32 10}

;--- test.ll
declare i32 @external(i32 %x)
define i32 @one() #0 !guid !0 {
  call void @llvm.instrprof.increment(ptr @one, i64 123, i32 1, i32 0)
  call void @llvm.instrprof.callsite(ptr @one, i64 123, i32 1, i32 0, ptr @external)
  %ret = call i32 @external(i32 1)
  ret i32 %ret
}

define i32 @caller(ptr %c) #1 !guid !1 {
  call void @llvm.instrprof.increment(ptr @caller, i64 567, i32 1, i32 0)
  call void @llvm.instrprof.callsite(ptr @caller, i64 567, i32 1, i32 0, ptr %c)
  %ret = call i32 %c()
  ret i32 %ret
}

define i32 @root(ptr %c) !guid !2 {
  call void @llvm.instrprof.increment(ptr @root, i64 432, i32 1, i32 0)
  call void @llvm.instrprof.callsite(ptr @root, i64 432, i32 2, i32 0, ptr @caller)
  %a = call i32 @caller(ptr %c)
  call void @llvm.instrprof.callsite(ptr @root, i64 432, i32 2, i32 1, ptr @caller)
  %b = call i32 @caller(ptr %c)
  %ret = add i32 %a, %b
  ret i32 %ret

}

attributes #0 = { alwaysinline }
attributes #1 = { noinline }
!0 = !{i64 1000}
!1 = !{i64 3000}
!2 = !{i64 4000}

;--- profile.json
[ {
  "Guid": 4000, "Counters":[10], "Callsites": [
    [{"Guid":3000, "Counters":[10], "Callsites":[[{"Guid":1000, "Counters":[10]}]]}],
    [{"Guid":3000, "Counters":[10], "Callsites":[[{"Guid":9000, "Counters":[10]}]]}]
  ]
}
]
