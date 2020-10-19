; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

; Do not outline calls to @llvm.eh.typeid.for. See llvm.org/PR39545.

@_ZTIi = external constant i8*

; CHECK-LABEL: @fun
; CHECK-NOT: call {{.*}}@fun.cold.1
define void @fun() {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ret void

if.else:
  %t = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  call void @sink()
  ret void
}

declare void @sink() cold

declare i32 @llvm.eh.typeid.for(i8*)
