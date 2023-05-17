; RUN: opt < %s -passes=instcombine -S | FileCheck %s

@.str254 = internal constant [2 x i8] c".\00"
@.str557 = internal constant [3 x i8] c"::\00"

define ptr @demangle_qualified(i32 %isfuncname) nounwind {
entry:
  %tobool272 = icmp ne i32 %isfuncname, 0
  %cond276 = select i1 %tobool272, ptr @.str254, ptr @.str557 ; <ptr> [#uses=4]
  %cmp.i504 = icmp eq ptr %cond276, null
  %rval = getelementptr i8, ptr %cond276, i1 %cmp.i504
  ret ptr %rval
}

; CHECK: %cond276 = select i1
; CHECK: ret ptr %cond276
