; RUN: opt < %s -passes=lower-switch

define void @test() {
	switch i32 0, label %Next [
	]
Next:		; preds = %0
	ret void
}

