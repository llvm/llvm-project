; RUN: llc -march=hexagon -mcpu=hexagonv5  < %s | FileCheck %s
; Check that we generate floating point compare in V5

; CHECK: p{{[0-2]+}} = sfcmp.{{.}}

define i32 @foo(float %y) nounwind {
entry:
  %retval = alloca i32, align 4
  %y.addr = alloca float, align 4
  store float %y, ptr %y.addr, align 4
  %0 = load float, ptr %y.addr, align 4
  %cmp = fcmp ogt float %0, 0x406AD7EFA0000000
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, ptr %retval
  br label %return

if.else:                                          ; preds = %entry
  store i32 2, ptr %retval
  br label %return

return:                                           ; preds = %if.else, %if.then
  %1 = load i32, ptr %retval
  ret i32 %1
}

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  %a = alloca float, align 4
  store i32 0, ptr %retval
  store float 0x40012E0A00000000, ptr %a, align 4
  %0 = load float, ptr %a, align 4
  %call = call i32 @foo(float %0)
  ret i32 %call
}
