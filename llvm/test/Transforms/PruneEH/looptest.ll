; RUN: opt < %s -prune-eh -S | FileCheck %s

declare void @nounwind() nounwind

define internal void @foo() {
	call void @nounwind()
	ret void
}

; CHECK-LABEL: @caller
define i32 @caller(i32 %n) personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit, !llvm.loop !0

body:
; CHECK: call void @foo(), !llvm.mem.parallel_loop_access !0
	invoke void @foo( )
			to label %Normal unwind label %Except, !llvm.mem.parallel_loop_access !0
  br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for, !llvm.loop !0

exit:
  br label %Normal

Normal:
	ret i32 0

Except:
        landingpad { i8*, i32 }
                catch i8* null
	ret i32 1
}

declare i32 @__gxx_personality_v0(...)

!0 = distinct !{!0}
