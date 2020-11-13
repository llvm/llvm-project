; RUN: opt < %s -prune-eh -enable-new-pm=0 -S | FileCheck %s
; RUN: opt < %s -passes='function-attrs,function(simplify-cfg)' -S | FileCheck %s

declare void @nounwind() nounwind

define internal void @foo() {
	call void @nounwind()
	ret void
}

; CHECK-LABEL: define i32 @caller()
define i32 @caller() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-NOT: invoke void @foo
	invoke void @foo( )
			to label %Normal unwind label %Except

Normal:		; preds = %0
	ret i32 0

Except:		; preds = %0
        landingpad { i8*, i32 }
                catch i8* null
	ret i32 1
}

declare i32 @__gxx_personality_v0(...)
