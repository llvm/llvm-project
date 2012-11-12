; RUN: opt < %s -instcombine -S | FileCheck %s

@.str254 = internal constant [2 x i8] c".\00"
@.str557 = internal constant [3 x i8] c"::\00"

define i8* @demangle_qualified(i32 %isfuncname) nounwind {
entry:
  %tobool272 = icmp ne i32 %isfuncname, 0
  %cond276 = select i1 %tobool272, i8* getelementptr inbounds ([2 x i8]* @.str254, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8]* @.str557, i32 0, i32 0) ; <i8*> [#uses=4]
  %cmp.i504 = icmp eq i8* %cond276, null
  %rval = getelementptr i8* %cond276, i1 %cmp.i504
  ret i8* %rval
}

; CHECK: %cond276 = select i1
; CHECK: ret i8* %cond276
