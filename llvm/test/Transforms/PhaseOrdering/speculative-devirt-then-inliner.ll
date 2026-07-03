; RUN: opt -S -O3 -enable-devirtualize-speculatively %s 2>&1 | FileCheck %s

; Test that the devirtualized calls are inlined.

@vt1 = constant [1 x ptr] [ptr @vf], !type !0
@vt2 = constant [1 x ptr] [ptr @vf2], !type !1


define i1 @vf(ptr %this) {
  ret i1 true
}

define i1 @vf2(ptr %this) {
  ret i1 false
}

; CHECK: define i1 @call
define i1 @call(ptr %obj) #1 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.public.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; if.true.direct_targ:                              ; preds = %0
  ;   br label %if.end.icp
  ; if.false.orig_indirect:                           ; preds = %0
  ;   %res = tail call i1 %fptr(ptr nonnull %obj)
  ;   br label %if.end.icp
  ; if.end.icp:                                       ; preds = %if.false.orig_indirect, %if.true.direct_targ
  ;   %2 = phi i1 [ %res, %if.false.orig_indirect ], [ true, %if.true.direct_targ ]
  ;   ret i1 %2
  %res = call i1 %fptr(ptr %obj)
  ret i1 %res
}


; CHECK: define i1 @call1
define i1 @call1(ptr %obj) #1 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable, align 8
  ; if.true.direct_targ:                              ; preds = %0
  ;   br label %if.end.icp
  ; if.false.orig_indirect:                           ; preds = %0
  ;   %res = tail call i1 %fptr(ptr nonnull %obj)
  ;   br label %if.end.icp
  ; if.end.icp:                                       ; preds = %if.false.orig_indirect, %if.true.direct_targ
  ;   %2 = phi i1 [ %res, %if.false.orig_indirect ], [ false, %if.true.direct_targ ]
  ;   ret i1 %2
  %res = call i1 %fptr(ptr %obj)
  ret i1 %res
}


declare i1 @llvm.type.test(ptr, metadata)
declare i1 @llvm.public.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
!1 = !{i32 0, !"typeid1"}
