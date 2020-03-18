; RUN: llc < %s -mtriple=x86_64-windows-msvc | FileCheck %s

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @foo() {
entry:
  %call = call i32 @cond()
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @abort1()
  unreachable

if.end:                                           ; preds = %entry
  %call1 = call i32 @cond()
  %tobool2 = icmp ne i32 %call1, 0
  br i1 %tobool2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.end
  call void @abort2()
  unreachable

if.end4:                                          ; preds = %if.end
  %call5 = call i32 @cond()
  %tobool6 = icmp ne i32 %call5, 0
  br i1 %tobool6, label %if.then7, label %if.end8

if.then7:                                         ; preds = %if.end4
  call void @abort3()
  unreachable

if.end8:                                          ; preds = %if.end4
  ret i32 0
}

; CHECK-LABEL: foo:
; CHECK: callq cond
; CHECK: callq cond
; CHECK: callq cond
;   We don't need int3's between these calls to abort, since they won't confuse
;   the unwinder.
; CHECK: callq abort1
; CHECK-NEXT:   # %if.then3
; CHECK: callq abort2
; CHECK-NEXT:   # %if.then7
; CHECK: callq abort3
; CHECK-NEXT: int3

declare dso_local i32 @cond()

declare dso_local void @abort1() noreturn
declare dso_local void @abort2() noreturn
declare dso_local void @abort3() noreturn
