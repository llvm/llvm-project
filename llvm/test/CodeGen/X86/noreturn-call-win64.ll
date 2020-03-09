; RUN: llc < %s -mtriple=x86_64-windows-msvc | FileCheck %s

%struct.MakeCleanup = type { i8 }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

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

define dso_local void @throw_exception() uwtable personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %o = alloca %struct.MakeCleanup, align 1
  %call = invoke i32 @cond()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %cmp1 = icmp eq i32 0, %call
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null)
          to label %unreachable unwind label %ehcleanup

if.end:                                           ; preds = %invoke.cont
  %call2 = invoke i32 @cond()
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %if.end
  %cmp2 = icmp eq i32 0, %call2
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %invoke.cont1
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null)
          to label %unreachable unwind label %ehcleanup

if.end4:                                          ; preds = %invoke.cont1
  call void @"??1MakeCleanup@@QEAA@XZ"(%struct.MakeCleanup* nonnull %o)
  ret void

ehcleanup:                                        ; preds = %if.then3, %if.end, %if.then, %entry
  %cp = cleanuppad within none []
  call void @"??1MakeCleanup@@QEAA@XZ"(%struct.MakeCleanup* nonnull %o) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller

unreachable:                                      ; preds = %if.then3, %if.then
  unreachable
}

declare dso_local i32 @__CxxFrameHandler3(...)
declare dso_local void @_CxxThrowException(i8*, %eh.ThrowInfo*)
declare dso_local void @"??1MakeCleanup@@QEAA@XZ"(%struct.MakeCleanup*)

; CHECK-LABEL: throw_exception:
; CHECK: callq cond
; CHECK: je
; CHECK: callq cond
; CHECK: je
; CHECK: retq
; CHECK: callq _CxxThrowException
; CHECK-NOT: {{(addq|subq) .*, %rsp}}
; CHECK: callq _CxxThrowException
; CHECK-NOT: {{(addq|subq) .*, %rsp}}
; CHECK: .seh_handlerdata
