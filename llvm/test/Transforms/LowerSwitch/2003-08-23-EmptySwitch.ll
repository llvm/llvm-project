; RUN: opt < %s -lowerswitch

define void @test() {
	switch i32 0, label %Next [
	]
Next:		; preds = %0
	ret void
}

