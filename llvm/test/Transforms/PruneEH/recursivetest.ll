; RUN: opt < %s -prune-eh -S | FileCheck %s
; RUN: opt < %s -passes='function-attrs,function(simplify-cfg)' -S | FileCheck %s

; CHECK-LABEL: define internal i32 @foo()
define internal i32 @foo() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-NOT: invoke i32 @foo()
	invoke i32 @foo( )
			to label %Normal unwind label %Except		; <i32>:1 [#uses=0]
Normal:		; preds = %0
	ret i32 12
Except:		; preds = %0
        landingpad { i8*, i32 }
                catch i8* null
	ret i32 123
}

; CHECK-LABEL: define i32 @caller()
define i32 @caller() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-NOT: invoke i32 @foo()
	invoke i32 @foo( )
			to label %Normal unwind label %Except		; <i32>:1 [#uses=0]
Normal:		; preds = %0
	ret i32 0
Except:		; preds = %0
        landingpad { i8*, i32 }
                catch i8* null
	ret i32 1
}

declare i32 @__gxx_personality_v0(...)
