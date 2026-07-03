; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

declare ptr @"personality_function"() #1
declare void @foo(i32)
declare void @bar() 
declare void @llvm.experimental.deoptimize.isVoid(...)
declare void @cold() cold

; Even though the likeliness of 'invoke' to throw an exception is assessed as low
; all other paths are even less likely. Check that hot paths leads to excepion handler.
define void @test1(i32 %0, i1 %arg) personality ptr @"personality_function"  !prof !1 {
;CHECK: edge %entry -> %unreached probability is 0x00000001 / 0x80000000 = 0.00%
;CHECK: edge %entry -> %invoke probability is 0x7fffffff / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge %invoke -> %invoke.cont.unreached probability is 0x00000000 / 0x80000000 = 0.00%
;CHECK: edge %invoke -> %land.pad probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge %land.pad -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

entry:
  br i1 %arg, label %unreached, label %invoke, !prof !2
invoke:
  invoke void @foo(i32 %0)
          to label %invoke.cont.unreached unwind label %land.pad
invoke.cont.unreached:
  call void (...) @llvm.experimental.deoptimize.isVoid(i32 10) [ "deopt"() ]
  ret void

unreached:
  unreachable

land.pad:
  %v20 = landingpad { ptr, i32 }
          cleanup
  %v21 = load ptr addrspace(1), ptr addrspace(256) inttoptr (i64 8 to ptr addrspace(256)), align 8
  br label %exit

exit:
  call void @bar()
  ret void
}

define void @test2(i32 %0, i1 %arg) personality ptr @"personality_function" {
;CHECK: edge %entry -> %unreached probability is 0x00000000 / 0x80000000 = 0.00%
;CHECK: edge %entry -> %invoke probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge %invoke -> %invoke.cont.cold probability is 0x7fff8000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge %invoke -> %land.pad probability is 0x00008000 / 0x80000000 = 0.00%
;CHECK: edge %land.pad -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

entry:
  br i1 %arg, label %unreached, label %invoke
invoke:
  invoke void @foo(i32 %0)
          to label %invoke.cont.cold unwind label %land.pad
invoke.cont.cold:
  call void @cold()
  ret void

unreached:
  unreachable

land.pad:
  %v20 = landingpad { ptr, i32 }
          cleanup
  %v21 = load ptr addrspace(1), ptr addrspace(256) inttoptr (i64 8 to ptr addrspace(256)), align 8
  br label %exit

exit:
  call void @bar()
  ret void
}

define void @test3(i32 %0, i1 %arg) personality ptr @"personality_function" {
;CHECK: edge %entry -> %unreached probability is 0x00000000 / 0x80000000 = 0.00%
;CHECK: edge %entry -> %invoke probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge %invoke -> %invoke.cont.cold probability is 0x7fff8000 / 0x80000000 = 100.00% [HOT edge]
;CHECK: edge %invoke -> %land.pad probability is 0x00008000 / 0x80000000 = 0.00%
;CHECK: edge %land.pad -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
entry:
  br i1 %arg, label %unreached, label %invoke
invoke:
  invoke void @foo(i32 %0)
          to label %invoke.cont.cold unwind label %land.pad
invoke.cont.cold:
  call void @cold()
  ret void

unreached:
  unreachable

land.pad:
  %v20 = landingpad { ptr, i32 }
          cleanup
  %v21 = load ptr addrspace(1), ptr addrspace(256) inttoptr (i64 8 to ptr addrspace(256)), align 8
  call void @cold()
  br label %exit

exit:
  call void @bar()
  ret void
}


attributes #1 = { nounwind }

!1 = !{!"function_entry_count", i64 32768}
!2 = !{!"branch_weights", i32 1, i32 983040}

