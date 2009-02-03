; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin

declare void @foo(i8*, i8*, i32, i32, i32, i32, i32, i32, i32)

define void @t() nounwind  {
	br label %1
; <label>:1		; preds = %0
	br label %bb4351.i
bb4351.i:		; preds = %1
	switch i32 0, label %bb4411.i [
		 i32 1, label %bb4354.i
		 i32 2, label %bb4369.i
	]
bb4354.i:		; preds = %bb4351.i
	br label %t.exit
bb4369.i:		; preds = %bb4351.i
	br label %bb4374.i
bb4374.i:		; preds = %bb4369.i
	br label %bb4411.i
bb4411.i:		; preds = %bb4374.i, %bb4351.i
	%sf4083.0.i = phi i32 [ 0, %bb4374.i ], [ 6, %bb4351.i ]		; <i32> [#uses=8]
	br label %bb4498.i
bb4498.i:		; preds = %bb4411.i
	%sfComp4077.1.i = phi i32 [ undef, %bb4411.i ]		; <i32> [#uses=2]
	%stComp4075.1.i = phi i32 [ undef, %bb4411.i ]		; <i32> [#uses=1]
	switch i32 0, label %bb4553.i [
		 i32 1, label %bb4501.i
		 i32 2, label %bb4521.i
	]
bb4501.i:		; preds = %bb4498.i
	%sfComp4077.1.reg2mem.0.i = phi i32 [ %sfComp4077.1.i, %bb4498.i ]		; <i32> [#uses=1]
	call void @foo( i8* null, i8* null, i32 %sfComp4077.1.reg2mem.0.i, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0 ) nounwind 
	br i1 false, label %UnifiedReturnBlock.i, label %bb4517.i
bb4517.i:		; preds = %bb4501.i
	br label %t.exit
bb4521.i:		; preds = %bb4498.i
	br label %bb4526.i
bb4526.i:		; preds = %bb4521.i
	switch i32 0, label %bb4529.i [
		 i32 6, label %bb4530.i
		 i32 7, label %bb4530.i
	]
bb4529.i:		; preds = %bb4526.i
	br label %bb4530.i
bb4530.i:		; preds = %bb4529.i, %bb4526.i, %bb4526.i
	br label %bb4553.i
bb4553.i:		; preds = %bb4530.i, %bb4498.i
	%dt4080.0.i = phi i32 [ %stComp4075.1.i, %bb4530.i ], [ 7, %bb4498.i ]		; <i32> [#uses=32]
	%df4081.0.i = phi i32 [ %sfComp4077.1.i, %bb4530.i ], [ 8, %bb4498.i ]		; <i32> [#uses=17]
	switch i32 %sf4083.0.i, label %bb4559.i [
		 i32 0, label %bb4558.i
		 i32 1, label %bb4558.i
		 i32 2, label %bb4558.i
		 i32 5, label %bb4561.i
		 i32 6, label %bb4561.i
		 i32 7, label %bb4561.i
		 i32 9, label %bb4557.i
	]
bb4557.i:		; preds = %bb4553.i
	switch i32 %df4081.0.i, label %bb4569.i [
		 i32 0, label %bb4568.i
		 i32 1, label %bb4568.i
		 i32 2, label %bb4568.i
		 i32 5, label %bb4571.i
		 i32 6, label %bb4571.i
		 i32 7, label %bb4571.i
		 i32 9, label %bb4567.i
	]
bb4558.i:		; preds = %bb4553.i, %bb4553.i, %bb4553.i
	switch i32 %df4081.0.i, label %bb4569.i [
		 i32 0, label %bb4568.i
		 i32 1, label %bb4568.i
		 i32 2, label %bb4568.i
		 i32 5, label %bb4571.i
		 i32 6, label %bb4571.i
		 i32 7, label %bb4571.i
		 i32 9, label %bb4567.i
	]
bb4559.i:		; preds = %bb4553.i
	br label %bb4561.i
bb4561.i:		; preds = %bb4559.i, %bb4553.i, %bb4553.i, %bb4553.i
	switch i32 %df4081.0.i, label %bb4569.i [
		 i32 0, label %bb4568.i
		 i32 1, label %bb4568.i
		 i32 2, label %bb4568.i
		 i32 5, label %bb4571.i
		 i32 6, label %bb4571.i
		 i32 7, label %bb4571.i
		 i32 9, label %bb4567.i
	]
bb4567.i:		; preds = %bb4561.i, %bb4558.i, %bb4557.i
	br label %bb4580.i
bb4568.i:		; preds = %bb4561.i, %bb4561.i, %bb4561.i, %bb4558.i, %bb4558.i, %bb4558.i, %bb4557.i, %bb4557.i, %bb4557.i
	br label %bb4580.i
bb4569.i:		; preds = %bb4561.i, %bb4558.i, %bb4557.i
	br label %bb4571.i
bb4571.i:		; preds = %bb4569.i, %bb4561.i, %bb4561.i, %bb4561.i, %bb4558.i, %bb4558.i, %bb4558.i, %bb4557.i, %bb4557.i, %bb4557.i
	br label %bb4580.i
bb4580.i:		; preds = %bb4571.i, %bb4568.i, %bb4567.i
	br i1 false, label %bb4611.i, label %bb4593.i
bb4593.i:		; preds = %bb4580.i
	br i1 false, label %bb4610.i, label %bb4611.i
bb4610.i:		; preds = %bb4593.i
	br label %bb4611.i
bb4611.i:		; preds = %bb4610.i, %bb4593.i, %bb4580.i
	br i1 false, label %bb4776.i, label %bb4620.i
bb4620.i:		; preds = %bb4611.i
	switch i32 0, label %bb4776.i [
		 i32 0, label %bb4691.i
		 i32 2, label %bb4740.i
		 i32 4, label %bb4755.i
		 i32 8, label %bb4622.i
		 i32 9, label %bb4622.i
		 i32 10, label %bb4629.i
		 i32 11, label %bb4629.i
		 i32 12, label %bb4651.i
		 i32 13, label %bb4651.i
		 i32 14, label %bb4665.i
		 i32 15, label %bb4665.i
		 i32 16, label %bb4691.i
		 i32 17, label %bb4691.i
		 i32 18, label %bb4712.i
		 i32 19, label %bb4712.i
		 i32 22, label %bb4733.i
		 i32 23, label %bb4733.i
	]
bb4622.i:		; preds = %bb4620.i, %bb4620.i
	br i1 false, label %bb4628.i, label %bb4776.i
bb4628.i:		; preds = %bb4622.i
	br label %bb4776.i
bb4629.i:		; preds = %bb4620.i, %bb4620.i
	br i1 false, label %bb4776.i, label %bb4644.i
bb4644.i:		; preds = %bb4629.i
	br i1 false, label %bb4650.i, label %bb4776.i
bb4650.i:		; preds = %bb4644.i
	br label %bb4776.i
bb4651.i:		; preds = %bb4620.i, %bb4620.i
	br i1 false, label %bb4776.i, label %bb4658.i
bb4658.i:		; preds = %bb4651.i
	br i1 false, label %bb4664.i, label %bb4776.i
bb4664.i:		; preds = %bb4658.i
	br label %bb4776.i
bb4665.i:		; preds = %bb4620.i, %bb4620.i
	br i1 false, label %bb4776.i, label %bb4684.i
bb4684.i:		; preds = %bb4665.i
	br i1 false, label %bb4690.i, label %bb4776.i
bb4690.i:		; preds = %bb4684.i
	br label %bb4776.i
bb4691.i:		; preds = %bb4620.i, %bb4620.i, %bb4620.i
	br i1 false, label %bb4776.i, label %bb4698.i
bb4698.i:		; preds = %bb4691.i
	br i1 false, label %bb4711.i, label %bb4776.i
bb4711.i:		; preds = %bb4698.i
	br label %bb4776.i
bb4712.i:		; preds = %bb4620.i, %bb4620.i
	br i1 false, label %bb4776.i, label %bb4726.i
bb4726.i:		; preds = %bb4712.i
	br i1 false, label %bb4732.i, label %bb4776.i
bb4732.i:		; preds = %bb4726.i
	br label %bb4776.i
bb4733.i:		; preds = %bb4620.i, %bb4620.i
	br i1 false, label %bb4739.i, label %bb4776.i
bb4739.i:		; preds = %bb4733.i
	br label %bb4776.i
bb4740.i:		; preds = %bb4620.i
	br i1 false, label %bb4776.i, label %bb4754.i
bb4754.i:		; preds = %bb4740.i
	br label %bb4776.i
bb4755.i:		; preds = %bb4620.i
	br i1 false, label %bb4776.i, label %bb4774.i
bb4774.i:		; preds = %bb4755.i
	br label %bb4776.i
bb4776.i:		; preds = %bb4774.i, %bb4755.i, %bb4754.i, %bb4740.i, %bb4739.i, %bb4733.i, %bb4732.i, %bb4726.i, %bb4712.i, %bb4711.i, %bb4698.i, %bb4691.i, %bb4690.i, %bb4684.i, %bb4665.i, %bb4664.i, %bb4658.i, %bb4651.i, %bb4650.i, %bb4644.i, %bb4629.i, %bb4628.i, %bb4622.i, %bb4620.i, %bb4611.i
	switch i32 0, label %bb4790.i [
		 i32 0, label %bb4786.i
		 i32 1, label %bb4784.i
		 i32 3, label %bb4784.i
		 i32 5, label %bb4784.i
		 i32 6, label %bb4785.i
		 i32 7, label %bb4785.i
		 i32 8, label %bb4791.i
		 i32 9, label %bb4791.i
		 i32 10, label %bb4791.i
		 i32 11, label %bb4791.i
		 i32 12, label %bb4791.i
		 i32 13, label %bb4791.i
		 i32 14, label %bb4791.i
		 i32 15, label %bb4791.i
		 i32 16, label %bb4791.i
		 i32 17, label %bb4791.i
		 i32 18, label %bb4791.i
		 i32 19, label %bb4791.i
	]
bb4784.i:		; preds = %bb4776.i, %bb4776.i, %bb4776.i
	br label %bb4791.i
bb4785.i:		; preds = %bb4776.i, %bb4776.i
	br label %bb4791.i
bb4786.i:		; preds = %bb4776.i
	br label %bb4791.i
bb4790.i:		; preds = %bb4776.i
	br label %bb4791.i
bb4791.i:		; preds = %bb4790.i, %bb4786.i, %bb4785.i, %bb4784.i, %bb4776.i, %bb4776.i, %bb4776.i, %bb4776.i, %bb4776.i, %bb4776.i, %bb4776.i, %bb4776.i, %bb4776.i, %bb4776.i, %bb4776.i, %bb4776.i
	switch i32 %dt4080.0.i, label %bb4803.i [
		 i32 0, label %bb4799.i
		 i32 6, label %bb4794.i
		 i32 7, label %bb4794.i
		 i32 8, label %bb4804.i
		 i32 9, label %bb4804.i
		 i32 10, label %bb4804.i
		 i32 11, label %bb4804.i
		 i32 12, label %bb4804.i
		 i32 13, label %bb4804.i
		 i32 14, label %bb4804.i
		 i32 15, label %bb4804.i
		 i32 16, label %bb4804.i
		 i32 17, label %bb4804.i
		 i32 18, label %bb4804.i
		 i32 19, label %bb4804.i
	]
bb4794.i:		; preds = %bb4791.i, %bb4791.i
	br i1 false, label %bb4809.i, label %bb4819.i
bb4799.i:		; preds = %bb4791.i
	br i1 false, label %bb4809.i, label %bb4819.i
bb4803.i:		; preds = %bb4791.i
	br label %bb4804.i
bb4804.i:		; preds = %bb4803.i, %bb4791.i, %bb4791.i, %bb4791.i, %bb4791.i, %bb4791.i, %bb4791.i, %bb4791.i, %bb4791.i, %bb4791.i, %bb4791.i, %bb4791.i, %bb4791.i
	br i1 false, label %bb4809.i, label %bb4819.i
bb4809.i:		; preds = %bb4804.i, %bb4799.i, %bb4794.i
	switch i32 %df4081.0.i, label %bb71.i.i [
		 i32 3, label %bb61.i.i
		 i32 4, label %bb.i.i
		 i32 5, label %bb.i.i
		 i32 6, label %bb.i.i
		 i32 7, label %bb.i.i
		 i32 8, label %bb38.i.i
		 i32 9, label %bb38.i.i
		 i32 10, label %bb50.i.i
		 i32 11, label %bb40.i.i
		 i32 16, label %bb38.i.i
	]
bb.i.i:		; preds = %bb4809.i, %bb4809.i, %bb4809.i, %bb4809.i
	br label %bb403.i.i
bb38.i.i:		; preds = %bb4809.i, %bb4809.i, %bb4809.i
	br label %bb403.i.i
bb40.i.i:		; preds = %bb4809.i
	br label %bb403.i.i
bb50.i.i:		; preds = %bb4809.i
	br label %bb403.i.i
bb61.i.i:		; preds = %bb4809.i
	br label %bb403.i.i
bb71.i.i:		; preds = %bb4809.i
	br label %bb403.i.i
bb403.i.i:		; preds = %bb71.i.i, %bb61.i.i, %bb50.i.i, %bb40.i.i, %bb38.i.i, %bb.i.i
	br i1 false, label %bb408.i.i, label %bb502.i.i
bb408.i.i:		; preds = %bb403.i.i
	br label %bb708.i.i
bb502.i.i:		; preds = %bb403.i.i
	br label %bb708.i.i
bb708.i.i:		; preds = %bb502.i.i, %bb408.i.i
	switch i32 0, label %bb758.i.i [
		 i32 0, label %bb710.i.i
		 i32 1, label %bb713.i.i
		 i32 2, label %bb718.i.i
		 i32 3, label %bb721.i.i
		 i32 4, label %bb726.i.i
		 i32 5, label %bb729.i.i
		 i32 8, label %bb732.i.i
		 i32 9, label %bb732.i.i
		 i32 10, label %bb737.i.i
		 i32 11, label %bb737.i.i
		 i32 12, label %bb742.i.i
		 i32 13, label %bb742.i.i
		 i32 14, label %bb745.i.i
		 i32 15, label %bb745.i.i
		 i32 16, label %bb750.i.i
		 i32 17, label %bb750.i.i
		 i32 18, label %bb753.i.i
		 i32 19, label %bb753.i.i
		 i32 22, label %bb750.i.i
		 i32 23, label %bb750.i.i
	]
bb710.i.i:		; preds = %bb708.i.i
	br label %bb758.i.i
bb713.i.i:		; preds = %bb708.i.i
	br label %bb758.i.i
bb718.i.i:		; preds = %bb708.i.i
	br label %bb758.i.i
bb721.i.i:		; preds = %bb708.i.i
	br label %bb758.i.i
bb726.i.i:		; preds = %bb708.i.i
	br label %bb758.i.i
bb729.i.i:		; preds = %bb708.i.i
	br label %bb758.i.i
bb732.i.i:		; preds = %bb708.i.i, %bb708.i.i
	br label %bb758.i.i
bb737.i.i:		; preds = %bb708.i.i, %bb708.i.i
	br label %bb758.i.i
bb742.i.i:		; preds = %bb708.i.i, %bb708.i.i
	br label %bb758.i.i
bb745.i.i:		; preds = %bb708.i.i, %bb708.i.i
	br label %bb758.i.i
bb750.i.i:		; preds = %bb708.i.i, %bb708.i.i, %bb708.i.i, %bb708.i.i
	br label %bb758.i.i
bb753.i.i:		; preds = %bb708.i.i, %bb708.i.i
	br label %bb758.i.i
bb758.i.i:		; preds = %bb753.i.i, %bb750.i.i, %bb745.i.i, %bb742.i.i, %bb737.i.i, %bb732.i.i, %bb729.i.i, %bb726.i.i, %bb721.i.i, %bb718.i.i, %bb713.i.i, %bb710.i.i, %bb708.i.i
	switch i32 %dt4080.0.i, label %bb808.i.i [
		 i32 0, label %bb760.i.i
		 i32 1, label %bb763.i.i
		 i32 2, label %bb768.i.i
		 i32 3, label %bb771.i.i
		 i32 4, label %bb776.i.i
		 i32 5, label %bb779.i.i
		 i32 8, label %bb782.i.i
		 i32 9, label %bb782.i.i
		 i32 10, label %bb787.i.i
		 i32 11, label %bb787.i.i
		 i32 12, label %bb792.i.i
		 i32 13, label %bb792.i.i
		 i32 14, label %bb795.i.i
		 i32 15, label %bb795.i.i
		 i32 16, label %bb800.i.i
		 i32 17, label %bb800.i.i
		 i32 18, label %bb803.i.i
		 i32 19, label %bb803.i.i
		 i32 22, label %bb800.i.i
		 i32 23, label %bb800.i.i
	]
bb760.i.i:		; preds = %bb758.i.i
	br label %bb811.i.i
bb763.i.i:		; preds = %bb758.i.i
	br label %bb811.i.i
bb768.i.i:		; preds = %bb758.i.i
	br label %bb811.i.i
bb771.i.i:		; preds = %bb758.i.i
	br label %bb811.i.i
bb776.i.i:		; preds = %bb758.i.i
	br label %bb811.i.i
bb779.i.i:		; preds = %bb758.i.i
	br label %bb811.i.i
bb782.i.i:		; preds = %bb758.i.i, %bb758.i.i
	br label %bb811.i.i
bb787.i.i:		; preds = %bb758.i.i, %bb758.i.i
	br label %bb811.i.i
bb792.i.i:		; preds = %bb758.i.i, %bb758.i.i
	br label %bb811.i.i
bb795.i.i:		; preds = %bb758.i.i, %bb758.i.i
	br label %bb811.i.i
bb800.i.i:		; preds = %bb758.i.i, %bb758.i.i, %bb758.i.i, %bb758.i.i
	br label %bb811.i.i
bb803.i.i:		; preds = %bb758.i.i, %bb758.i.i
	br label %bb808.i.i
bb808.i.i:		; preds = %bb803.i.i, %bb758.i.i
	br label %bb811.i.i
bb811.i.i:		; preds = %bb808.i.i, %bb800.i.i, %bb795.i.i, %bb792.i.i, %bb787.i.i, %bb782.i.i, %bb779.i.i, %bb776.i.i, %bb771.i.i, %bb768.i.i, %bb763.i.i, %bb760.i.i
	switch i32 0, label %bb928.i.i [
		 i32 0, label %bb813.i.i
		 i32 1, label %bb833.i.i
		 i32 2, label %bb813.i.i
		 i32 3, label %bb833.i.i
		 i32 4, label %bb813.i.i
		 i32 5, label %bb813.i.i
		 i32 8, label %bb872.i.i
		 i32 9, label %bb872.i.i
		 i32 10, label %bb890.i.i
		 i32 11, label %bb890.i.i
		 i32 12, label %bb813.i.i
		 i32 13, label %bb813.i.i
		 i32 14, label %bb908.i.i
		 i32 15, label %bb908.i.i
		 i32 16, label %bb813.i.i
		 i32 17, label %bb813.i.i
		 i32 18, label %bb908.i.i
		 i32 19, label %bb908.i.i
		 i32 22, label %bb813.i.i
		 i32 23, label %bb813.i.i
	]
bb813.i.i:		; preds = %bb811.i.i, %bb811.i.i, %bb811.i.i, %bb811.i.i, %bb811.i.i, %bb811.i.i, %bb811.i.i, %bb811.i.i, %bb811.i.i, %bb811.i.i
	switch i32 %dt4080.0.i, label %bb1065.i.i [
		 i32 0, label %bb930.i.i
		 i32 1, label %bb950.i.i
		 i32 2, label %bb930.i.i
		 i32 3, label %bb950.i.i
		 i32 4, label %bb989.i.i
		 i32 5, label %bb989.i.i
		 i32 8, label %bb1009.i.i
		 i32 9, label %bb1009.i.i
		 i32 10, label %bb1027.i.i
		 i32 11, label %bb1027.i.i
		 i32 12, label %bb930.i.i
		 i32 13, label %bb930.i.i
		 i32 14, label %bb1045.i.i
		 i32 15, label %bb1045.i.i
		 i32 16, label %bb930.i.i
		 i32 17, label %bb930.i.i
		 i32 18, label %bb1045.i.i
		 i32 19, label %bb1045.i.i
		 i32 22, label %bb930.i.i
		 i32 23, label %bb930.i.i
	]
bb833.i.i:		; preds = %bb811.i.i, %bb811.i.i
	switch i32 %dt4080.0.i, label %bb1065.i.i [
		 i32 0, label %bb930.i.i
		 i32 1, label %bb950.i.i
		 i32 2, label %bb930.i.i
		 i32 3, label %bb950.i.i
		 i32 4, label %bb989.i.i
		 i32 5, label %bb989.i.i
		 i32 8, label %bb1009.i.i
		 i32 9, label %bb1009.i.i
		 i32 10, label %bb1027.i.i
		 i32 11, label %bb1027.i.i
		 i32 12, label %bb930.i.i
		 i32 13, label %bb930.i.i
		 i32 14, label %bb1045.i.i
		 i32 15, label %bb1045.i.i
		 i32 16, label %bb930.i.i
		 i32 17, label %bb930.i.i
		 i32 18, label %bb1045.i.i
		 i32 19, label %bb1045.i.i
		 i32 22, label %bb930.i.i
		 i32 23, label %bb930.i.i
	]
bb872.i.i:		; preds = %bb811.i.i, %bb811.i.i
	switch i32 %dt4080.0.i, label %bb1065.i.i [
		 i32 0, label %bb930.i.i
		 i32 1, label %bb950.i.i
		 i32 2, label %bb930.i.i
		 i32 3, label %bb950.i.i
		 i32 4, label %bb989.i.i
		 i32 5, label %bb989.i.i
		 i32 8, label %bb1009.i.i
		 i32 9, label %bb1009.i.i
		 i32 10, label %bb1027.i.i
		 i32 11, label %bb1027.i.i
		 i32 12, label %bb930.i.i
		 i32 13, label %bb930.i.i
		 i32 14, label %bb1045.i.i
		 i32 15, label %bb1045.i.i
		 i32 16, label %bb930.i.i
		 i32 17, label %bb930.i.i
		 i32 18, label %bb1045.i.i
		 i32 19, label %bb1045.i.i
		 i32 22, label %bb930.i.i
		 i32 23, label %bb930.i.i
	]
bb890.i.i:		; preds = %bb811.i.i, %bb811.i.i
	switch i32 %dt4080.0.i, label %bb1065.i.i [
		 i32 0, label %bb930.i.i
		 i32 1, label %bb950.i.i
		 i32 2, label %bb930.i.i
		 i32 3, label %bb950.i.i
		 i32 4, label %bb989.i.i
		 i32 5, label %bb989.i.i
		 i32 8, label %bb1009.i.i
		 i32 9, label %bb1009.i.i
		 i32 10, label %bb1027.i.i
		 i32 11, label %bb1027.i.i
		 i32 12, label %bb930.i.i
		 i32 13, label %bb930.i.i
		 i32 14, label %bb1045.i.i
		 i32 15, label %bb1045.i.i
		 i32 16, label %bb930.i.i
		 i32 17, label %bb930.i.i
		 i32 18, label %bb1045.i.i
		 i32 19, label %bb1045.i.i
		 i32 22, label %bb930.i.i
		 i32 23, label %bb930.i.i
	]
bb908.i.i:		; preds = %bb811.i.i, %bb811.i.i, %bb811.i.i, %bb811.i.i
	br label %bb928.i.i
bb928.i.i:		; preds = %bb908.i.i, %bb811.i.i
	switch i32 %dt4080.0.i, label %bb1065.i.i [
		 i32 0, label %bb930.i.i
		 i32 1, label %bb950.i.i
		 i32 2, label %bb930.i.i
		 i32 3, label %bb950.i.i
		 i32 4, label %bb989.i.i
		 i32 5, label %bb989.i.i
		 i32 8, label %bb1009.i.i
		 i32 9, label %bb1009.i.i
		 i32 10, label %bb1027.i.i
		 i32 11, label %bb1027.i.i
		 i32 12, label %bb930.i.i
		 i32 13, label %bb930.i.i
		 i32 14, label %bb1045.i.i
		 i32 15, label %bb1045.i.i
		 i32 16, label %bb930.i.i
		 i32 17, label %bb930.i.i
		 i32 18, label %bb1045.i.i
		 i32 19, label %bb1045.i.i
		 i32 22, label %bb930.i.i
		 i32 23, label %bb930.i.i
	]
bb930.i.i:		; preds = %bb928.i.i, %bb928.i.i, %bb928.i.i, %bb928.i.i, %bb928.i.i, %bb928.i.i, %bb928.i.i, %bb928.i.i, %bb890.i.i, %bb890.i.i, %bb890.i.i, %bb890.i.i, %bb890.i.i, %bb890.i.i, %bb890.i.i, %bb890.i.i, %bb872.i.i, %bb872.i.i, %bb872.i.i, %bb872.i.i, %bb872.i.i, %bb872.i.i, %bb872.i.i, %bb872.i.i, %bb833.i.i, %bb833.i.i, %bb833.i.i, %bb833.i.i, %bb833.i.i, %bb833.i.i, %bb833.i.i, %bb833.i.i, %bb813.i.i, %bb813.i.i, %bb813.i.i, %bb813.i.i, %bb813.i.i, %bb813.i.i, %bb813.i.i, %bb813.i.i
	br label %bb5235.i
bb950.i.i:		; preds = %bb928.i.i, %bb928.i.i, %bb890.i.i, %bb890.i.i, %bb872.i.i, %bb872.i.i, %bb833.i.i, %bb833.i.i, %bb813.i.i, %bb813.i.i
	br label %bb5235.i
bb989.i.i:		; preds = %bb928.i.i, %bb928.i.i, %bb890.i.i, %bb890.i.i, %bb872.i.i, %bb872.i.i, %bb833.i.i, %bb833.i.i, %bb813.i.i, %bb813.i.i
	br label %bb5235.i
bb1009.i.i:		; preds = %bb928.i.i, %bb928.i.i, %bb890.i.i, %bb890.i.i, %bb872.i.i, %bb872.i.i, %bb833.i.i, %bb833.i.i, %bb813.i.i, %bb813.i.i
	br label %bb5235.i
bb1027.i.i:		; preds = %bb928.i.i, %bb928.i.i, %bb890.i.i, %bb890.i.i, %bb872.i.i, %bb872.i.i, %bb833.i.i, %bb833.i.i, %bb813.i.i, %bb813.i.i
	br label %bb5235.i
bb1045.i.i:		; preds = %bb928.i.i, %bb928.i.i, %bb928.i.i, %bb928.i.i, %bb890.i.i, %bb890.i.i, %bb890.i.i, %bb890.i.i, %bb872.i.i, %bb872.i.i, %bb872.i.i, %bb872.i.i, %bb833.i.i, %bb833.i.i, %bb833.i.i, %bb833.i.i, %bb813.i.i, %bb813.i.i, %bb813.i.i, %bb813.i.i
	br label %bb1065.i.i
bb1065.i.i:		; preds = %bb1045.i.i, %bb928.i.i, %bb890.i.i, %bb872.i.i, %bb833.i.i, %bb813.i.i
	br label %bb5235.i
bb4819.i:		; preds = %bb4804.i, %bb4799.i, %bb4794.i
	br i1 false, label %bb5208.i, label %bb5011.i
bb5011.i:		; preds = %bb4819.i
	switch i32 0, label %bb5039.i [
		 i32 10, label %bb5016.i
		 i32 3, label %bb5103.i
	]
bb5016.i:		; preds = %bb5011.i
	br i1 false, label %bb5103.i, label %bb5039.i
bb5039.i:		; preds = %bb5016.i, %bb5011.i
	switch i32 0, label %bb5052.i [
		 i32 3, label %bb5103.i
		 i32 10, label %bb5103.i
	]
bb5052.i:		; preds = %bb5039.i
	br i1 false, label %bb5103.i, label %bb5065.i
bb5065.i:		; preds = %bb5052.i
	br i1 false, label %bb5078.i, label %bb5103.i
bb5078.i:		; preds = %bb5065.i
	br i1 false, label %bb5103.i, label %bb5084.i
bb5084.i:		; preds = %bb5078.i
	br i1 false, label %bb5103.i, label %bb5090.i
bb5090.i:		; preds = %bb5084.i
	br i1 false, label %bb5103.i, label %bb5096.i
bb5096.i:		; preds = %bb5090.i
	br i1 false, label %bb5103.i, label %bb5102.i
bb5102.i:		; preds = %bb5096.i
	br label %bb5103.i
bb5103.i:		; preds = %bb5102.i, %bb5096.i, %bb5090.i, %bb5084.i, %bb5078.i, %bb5065.i, %bb5052.i, %bb5039.i, %bb5039.i, %bb5016.i, %bb5011.i
	switch i32 0, label %bb5208.i [
		 i32 0, label %bb5133.i
		 i32 2, label %bb5162.i
		 i32 4, label %bb5182.i
		 i32 10, label %bb5113.i
		 i32 11, label %bb5113.i
		 i32 12, label %bb5121.i
		 i32 13, label %bb5121.i
		 i32 14, label %bb5125.i
		 i32 15, label %bb5125.i
		 i32 16, label %bb5133.i
		 i32 17, label %bb5133.i
		 i32 18, label %bb5146.i
		 i32 19, label %bb5146.i
	]
bb5113.i:		; preds = %bb5103.i, %bb5103.i
	switch i32 %dt4080.0.i, label %bb5208.i [
		 i32 8, label %bb5115.i
		 i32 9, label %bb5115.i
		 i32 12, label %bb5117.i
		 i32 13, label %bb5117.i
		 i32 14, label %bb5119.i
		 i32 15, label %bb5119.i
	]
bb5115.i:		; preds = %bb5113.i, %bb5113.i
	br label %bb5208.i
bb5117.i:		; preds = %bb5113.i, %bb5113.i
	br label %bb5208.i
bb5119.i:		; preds = %bb5113.i, %bb5113.i
	br label %bb5208.i
bb5121.i:		; preds = %bb5103.i, %bb5103.i
	switch i32 %dt4080.0.i, label %bb5208.i [
		 i32 8, label %bb5123.i
		 i32 9, label %bb5123.i
	]
bb5123.i:		; preds = %bb5121.i, %bb5121.i
	br label %bb5208.i
bb5125.i:		; preds = %bb5103.i, %bb5103.i
	switch i32 %dt4080.0.i, label %bb5208.i [
		 i32 8, label %bb5127.i
		 i32 9, label %bb5127.i
		 i32 12, label %bb5129.i
		 i32 13, label %bb5129.i
	]
bb5127.i:		; preds = %bb5125.i, %bb5125.i
	br label %bb5208.i
bb5129.i:		; preds = %bb5125.i, %bb5125.i
	br label %bb5208.i
bb5133.i:		; preds = %bb5103.i, %bb5103.i, %bb5103.i
	switch i32 %dt4080.0.i, label %bb5208.i [
		 i32 8, label %bb5135.i
		 i32 9, label %bb5135.i
		 i32 10, label %bb5137.i
		 i32 11, label %bb5137.i
		 i32 12, label %bb5139.i
		 i32 13, label %bb5139.i
		 i32 14, label %bb5143.i
		 i32 15, label %bb5143.i
	]
bb5135.i:		; preds = %bb5133.i, %bb5133.i
	br label %bb5208.i
bb5137.i:		; preds = %bb5133.i, %bb5133.i
	br label %bb5208.i
bb5139.i:		; preds = %bb5133.i, %bb5133.i
	br label %bb5208.i
bb5143.i:		; preds = %bb5133.i, %bb5133.i
	br label %bb5208.i
bb5146.i:		; preds = %bb5103.i, %bb5103.i
	switch i32 %dt4080.0.i, label %bb5208.i [
		 i32 0, label %bb5158.i
		 i32 8, label %bb5148.i
		 i32 9, label %bb5148.i
		 i32 10, label %bb5150.i
		 i32 11, label %bb5150.i
		 i32 12, label %bb5152.i
		 i32 13, label %bb5152.i
		 i32 14, label %bb5155.i
		 i32 15, label %bb5155.i
		 i32 16, label %bb5158.i
		 i32 17, label %bb5158.i
	]
bb5148.i:		; preds = %bb5146.i, %bb5146.i
	br label %bb5208.i
bb5150.i:		; preds = %bb5146.i, %bb5146.i
	br label %bb5208.i
bb5152.i:		; preds = %bb5146.i, %bb5146.i
	br label %bb5208.i
bb5155.i:		; preds = %bb5146.i, %bb5146.i
	br label %bb5208.i
bb5158.i:		; preds = %bb5146.i, %bb5146.i, %bb5146.i
	br label %bb5208.i
bb5162.i:		; preds = %bb5103.i
	switch i32 %dt4080.0.i, label %bb5208.i [
		 i32 0, label %bb5175.i
		 i32 8, label %bb5164.i
		 i32 9, label %bb5164.i
		 i32 10, label %bb5166.i
		 i32 11, label %bb5166.i
		 i32 12, label %bb5168.i
		 i32 13, label %bb5168.i
		 i32 14, label %bb5172.i
		 i32 15, label %bb5172.i
		 i32 16, label %bb5175.i
		 i32 17, label %bb5175.i
		 i32 18, label %bb5179.i
		 i32 19, label %bb5179.i
	]
bb5164.i:		; preds = %bb5162.i, %bb5162.i
	br label %bb5208.i
bb5166.i:		; preds = %bb5162.i, %bb5162.i
	br label %bb5208.i
bb5168.i:		; preds = %bb5162.i, %bb5162.i
	br label %bb5208.i
bb5172.i:		; preds = %bb5162.i, %bb5162.i
	br label %bb5208.i
bb5175.i:		; preds = %bb5162.i, %bb5162.i, %bb5162.i
	br label %bb5208.i
bb5179.i:		; preds = %bb5162.i, %bb5162.i
	br label %bb5208.i
bb5182.i:		; preds = %bb5103.i
	switch i32 %dt4080.0.i, label %bb5208.i [
		 i32 0, label %bb5195.i
		 i32 2, label %bb5202.i
		 i32 8, label %bb5184.i
		 i32 9, label %bb5184.i
		 i32 10, label %bb5186.i
		 i32 11, label %bb5186.i
		 i32 12, label %bb5188.i
		 i32 13, label %bb5188.i
		 i32 14, label %bb5192.i
		 i32 15, label %bb5192.i
		 i32 16, label %bb5195.i
		 i32 17, label %bb5195.i
		 i32 18, label %bb5199.i
		 i32 19, label %bb5199.i
	]
bb5184.i:		; preds = %bb5182.i, %bb5182.i
	br label %bb5208.i
bb5186.i:		; preds = %bb5182.i, %bb5182.i
	br label %bb5208.i
bb5188.i:		; preds = %bb5182.i, %bb5182.i
	br label %bb5208.i
bb5192.i:		; preds = %bb5182.i, %bb5182.i
	br label %bb5208.i
bb5195.i:		; preds = %bb5182.i, %bb5182.i, %bb5182.i
	br label %bb5208.i
bb5199.i:		; preds = %bb5182.i, %bb5182.i
	br label %bb5208.i
bb5202.i:		; preds = %bb5182.i
	br label %bb5208.i
bb5208.i:		; preds = %bb5202.i, %bb5199.i, %bb5195.i, %bb5192.i, %bb5188.i, %bb5186.i, %bb5184.i, %bb5182.i, %bb5179.i, %bb5175.i, %bb5172.i, %bb5168.i, %bb5166.i, %bb5164.i, %bb5162.i, %bb5158.i, %bb5155.i, %bb5152.i, %bb5150.i, %bb5148.i, %bb5146.i, %bb5143.i, %bb5139.i, %bb5137.i, %bb5135.i, %bb5133.i, %bb5129.i, %bb5127.i, %bb5125.i, %bb5123.i, %bb5121.i, %bb5119.i, %bb5117.i, %bb5115.i, %bb5113.i, %bb5103.i, %bb4819.i
	switch i32 0, label %bb5221.i [
		 i32 0, label %bb5210.i
		 i32 1, label %bb5211.i
		 i32 2, label %bb5212.i
		 i32 3, label %bb5213.i
		 i32 4, label %bb5214.i
		 i32 5, label %bb5215.i
		 i32 6, label %bb5217.i
		 i32 7, label %bb5216.i
		 i32 12, label %bb5218.i
		 i32 13, label %bb5218.i
		 i32 14, label %bb5219.i
		 i32 15, label %bb5219.i
		 i32 16, label %bb5210.i
		 i32 17, label %bb5210.i
		 i32 22, label %bb5210.i
		 i32 23, label %bb5210.i
	]
bb5210.i:		; preds = %bb5208.i, %bb5208.i, %bb5208.i, %bb5208.i, %bb5208.i
	br label %bb5224.i
bb5211.i:		; preds = %bb5208.i
	br label %bb5224.i
bb5212.i:		; preds = %bb5208.i
	br label %bb5224.i
bb5213.i:		; preds = %bb5208.i
	br label %bb5224.i
bb5214.i:		; preds = %bb5208.i
	br label %bb5224.i
bb5215.i:		; preds = %bb5208.i
	br label %bb5224.i
bb5216.i:		; preds = %bb5208.i
	br label %bb5224.i
bb5217.i:		; preds = %bb5208.i
	br label %bb5224.i
bb5218.i:		; preds = %bb5208.i, %bb5208.i
	br label %bb5224.i
bb5219.i:		; preds = %bb5208.i, %bb5208.i
	br label %bb5224.i
bb5221.i:		; preds = %bb5208.i
	br label %bb5224.i
bb5224.i:		; preds = %bb5221.i, %bb5219.i, %bb5218.i, %bb5217.i, %bb5216.i, %bb5215.i, %bb5214.i, %bb5213.i, %bb5212.i, %bb5211.i, %bb5210.i
	br label %bb5235.i
bb5235.i:		; preds = %bb5224.i, %bb1065.i.i, %bb1027.i.i, %bb1009.i.i, %bb989.i.i, %bb950.i.i, %bb930.i.i
	br label %bb5272.i
bb5272.i:		; preds = %bb5235.i
	br label %bb5276.i
bb5276.i:		; preds = %bb19808.i, %bb5272.i
	br label %bb16607.i
bb5295.i:		; preds = %bb5295.preheader.i, %storeVecColor_RGB_UI.exit
	br label %loadVecColor_BGRA_UI8888R.exit
loadVecColor_BGRA_UI8888R.exit:		; preds = %bb5295.i
	br i1 false, label %bb5325.i, label %bb5351.i
bb5325.i:		; preds = %loadVecColor_BGRA_UI8888R.exit
	br i1 false, label %bb4527.i, label %bb.i
bb.i:		; preds = %bb5325.i
	switch i32 0, label %bb4527.i [
		 i32 4, label %bb4362.i
		 i32 8, label %bb4448.i
	]
bb4362.i:		; preds = %bb.i
	br i1 false, label %bb4532.i, label %bb5556.i
bb4448.i:		; preds = %bb.i
	br label %bb4527.i
bb4527.i:		; preds = %bb4448.i, %bb.i, %bb5325.i
	br i1 false, label %bb4532.i, label %bb5556.i
bb4532.i:		; preds = %bb4527.i, %bb4362.i
	switch i32 0, label %bb4997.i [
		 i32 6, label %bb4534.i
		 i32 7, label %bb4982.i
	]
bb4534.i:		; preds = %bb4532.i
	br i1 false, label %bb4875.i, label %bb4619.i
bb4619.i:		; preds = %bb4534.i
	br i1 false, label %bb4875.i, label %bb4663.i
bb4663.i:		; preds = %bb4619.i
	br label %bb4855.i
bb4759.i:		; preds = %bb4855.i
	br label %bb4855.i
bb4855.i:		; preds = %bb4759.i, %bb4663.i
	br i1 false, label %bb4866.i, label %bb4759.i
bb4866.i:		; preds = %bb4855.i
	br label %bb4875.i
bb4875.i:		; preds = %bb4866.i, %bb4619.i, %bb4534.i
	br i1 false, label %bb4973.i, label %bb4922.i
bb4922.i:		; preds = %bb4875.i
	br label %bb4973.i
bb4973.i:		; preds = %bb4922.i, %bb4875.i
	br label %bb4982.i
bb4982.i:		; preds = %bb4973.i, %bb4532.i
	br label %bb5041.i
bb4997.i:		; preds = %bb4532.i
	br label %bb5041.i
bb5041.i:		; preds = %bb4997.i, %bb4982.i
	switch i32 0, label %bb5464.i [
		 i32 0, label %bb5344.i
		 i32 1, label %bb5374.i
		 i32 2, label %bb5404.i
		 i32 3, label %bb5434.i
		 i32 11, label %bb5263.i
	]
bb5263.i:		; preds = %bb5041.i
	br i1 false, label %bb12038.i, label %bb5467.i
bb5344.i:		; preds = %bb5041.i
	br i1 false, label %bb12038.i, label %bb5467.i
bb5374.i:		; preds = %bb5041.i
	br i1 false, label %bb12038.i, label %bb5467.i
bb5404.i:		; preds = %bb5041.i
	br i1 false, label %bb12038.i, label %bb5467.i
bb5434.i:		; preds = %bb5041.i
	br label %bb5464.i
bb5464.i:		; preds = %bb5434.i, %bb5041.i
	br i1 false, label %bb12038.i, label %bb5467.i
bb5467.i:		; preds = %bb5464.i, %bb5404.i, %bb5374.i, %bb5344.i, %bb5263.i
	switch i32 0, label %bb15866.i [
		 i32 3, label %bb13016.i
		 i32 4, label %bb12040.i
		 i32 8, label %bb12514.i
		 i32 10, label %bb12903.i
		 i32 11, label %bb12553.i
		 i32 16, label %bb12514.i
	]
bb5556.i:		; preds = %bb4527.i, %bb4362.i
	switch i32 0, label %bb8990.i [
		 i32 3, label %bb6403.i
		 i32 4, label %bb6924.i
		 i32 8, label %bb6924.i
		 i32 10, label %bb6403.i
		 i32 11, label %bb5882.i
		 i32 16, label %bb5558.i
	]
bb5558.i:		; preds = %bb5556.i
	br label %bb8990.i
bb5882.i:		; preds = %bb5556.i
	switch i32 0, label %bb6387.i [
		 i32 1, label %bb6332.i
		 i32 3, label %bb6332.i
		 i32 4, label %bb6352.i
		 i32 6, label %bb5884.i
		 i32 7, label %bb8990.i
	]
bb5884.i:		; preds = %bb5882.i
	br i1 false, label %bb6225.i, label %bb5969.i
bb5969.i:		; preds = %bb5884.i
	br i1 false, label %bb6225.i, label %bb6013.i
bb6013.i:		; preds = %bb5969.i
	br label %bb6205.i
bb6109.i:		; preds = %bb6205.i
	br label %bb6205.i
bb6205.i:		; preds = %bb6109.i, %bb6013.i
	br i1 false, label %bb6216.i, label %bb6109.i
bb6216.i:		; preds = %bb6205.i
	br label %bb6225.i
bb6225.i:		; preds = %bb6216.i, %bb5969.i, %bb5884.i
	br i1 false, label %bb6323.i, label %bb6272.i
bb6272.i:		; preds = %bb6225.i
	switch i32 0, label %bb6908.i [
		 i32 1, label %bb6853.i48
		 i32 3, label %bb6853.i48
		 i32 4, label %bb6873.i
		 i32 6, label %bb6405.i
		 i32 7, label %bb8990.i
	]
bb6323.i:		; preds = %bb6225.i
	switch i32 0, label %bb6908.i [
		 i32 1, label %bb6853.i48
		 i32 3, label %bb6853.i48
		 i32 4, label %bb6873.i
		 i32 6, label %bb6405.i
		 i32 7, label %bb8990.i
	]
bb6332.i:		; preds = %bb5882.i, %bb5882.i
	switch i32 0, label %bb6908.i [
		 i32 1, label %bb6853.i48
		 i32 3, label %bb6853.i48
		 i32 4, label %bb6873.i
		 i32 6, label %bb6405.i
		 i32 7, label %bb8990.i
	]
bb6352.i:		; preds = %bb5882.i
	br label %bb6873.i
bb6387.i:		; preds = %bb5882.i
	br label %bb6403.i
bb6403.i:		; preds = %bb6387.i, %bb5556.i, %bb5556.i
	switch i32 0, label %bb6908.i [
		 i32 1, label %bb6853.i48
		 i32 3, label %bb6853.i48
		 i32 4, label %bb6873.i
		 i32 6, label %bb6405.i
		 i32 7, label %bb8990.i
	]
bb6405.i:		; preds = %bb6403.i, %bb6332.i, %bb6323.i, %bb6272.i
	br i1 false, label %bb6746.i, label %bb6490.i
bb6490.i:		; preds = %bb6405.i
	br i1 false, label %bb6746.i, label %bb6534.i
bb6534.i:		; preds = %bb6490.i
	br label %bb6726.i
bb6630.i:		; preds = %bb6726.i
	br label %bb6726.i
bb6726.i:		; preds = %bb6630.i, %bb6534.i
	br i1 false, label %bb6737.i, label %bb6630.i
bb6737.i:		; preds = %bb6726.i
	br label %bb6746.i
bb6746.i:		; preds = %bb6737.i, %bb6490.i, %bb6405.i
	br i1 false, label %bb6844.i, label %bb6793.i
bb6793.i:		; preds = %bb6746.i
	br label %bb8990.i
bb6844.i:		; preds = %bb6746.i
	br label %bb8990.i
bb6853.i48:		; preds = %bb6403.i, %bb6403.i, %bb6332.i, %bb6332.i, %bb6323.i, %bb6323.i, %bb6272.i, %bb6272.i
	br label %bb8990.i
bb6873.i:		; preds = %bb6403.i, %bb6352.i, %bb6332.i, %bb6323.i, %bb6272.i
	br label %bb8990.i
bb6908.i:		; preds = %bb6403.i, %bb6332.i, %bb6323.i, %bb6272.i
	br label %bb8990.i
bb6924.i:		; preds = %bb5556.i, %bb5556.i
	switch i32 0, label %bb8929.i [
		 i32 1, label %bb8715.i
		 i32 3, label %bb8715.i
		 i32 4, label %bb8792.i
		 i32 6, label %bb6926.i
		 i32 7, label %bb8990.i
	]
bb6926.i:		; preds = %bb6924.i
	br i1 false, label %bb7267.i, label %bb7011.i
bb7011.i:		; preds = %bb6926.i
	br i1 false, label %bb7267.i, label %bb7055.i
bb7055.i:		; preds = %bb7011.i
	br label %bb7247.i
bb7151.i:		; preds = %bb7247.i
	br label %bb7247.i
bb7247.i:		; preds = %bb7151.i, %bb7055.i
	br i1 false, label %bb7258.i, label %bb7151.i
bb7258.i:		; preds = %bb7247.i
	br label %bb7267.i
bb7267.i:		; preds = %bb7258.i, %bb7011.i, %bb6926.i
	br i1 false, label %bb7365.i, label %bb7314.i
bb7314.i:		; preds = %bb7267.i
	br label %bb7365.i
bb7365.i:		; preds = %bb7314.i, %bb7267.i
	br i1 false, label %bb7714.i, label %bb7458.i
bb7458.i:		; preds = %bb7365.i
	br i1 false, label %bb7714.i, label %bb7502.i
bb7502.i:		; preds = %bb7458.i
	br label %bb7694.i
bb7598.i:		; preds = %bb7694.i
	br label %bb7694.i
bb7694.i:		; preds = %bb7598.i, %bb7502.i
	br i1 false, label %bb7705.i, label %bb7598.i
bb7705.i:		; preds = %bb7694.i
	br label %bb7714.i
bb7714.i:		; preds = %bb7705.i, %bb7458.i, %bb7365.i
	br i1 false, label %bb7812.i, label %bb7761.i
bb7761.i:		; preds = %bb7714.i
	br label %bb7812.i
bb7812.i:		; preds = %bb7761.i, %bb7714.i
	br i1 false, label %bb8161.i, label %bb7905.i
bb7905.i:		; preds = %bb7812.i
	br i1 false, label %bb8161.i, label %bb7949.i
bb7949.i:		; preds = %bb7905.i
	br label %bb8141.i
bb8045.i:		; preds = %bb8141.i
	br label %bb8141.i
bb8141.i:		; preds = %bb8045.i, %bb7949.i
	br i1 false, label %bb8152.i, label %bb8045.i
bb8152.i:		; preds = %bb8141.i
	br label %bb8161.i
bb8161.i:		; preds = %bb8152.i, %bb7905.i, %bb7812.i
	br i1 false, label %bb8259.i, label %bb8208.i
bb8208.i:		; preds = %bb8161.i
	br label %bb8259.i
bb8259.i:		; preds = %bb8208.i, %bb8161.i
	br i1 false, label %bb8608.i, label %bb8352.i
bb8352.i:		; preds = %bb8259.i
	br i1 false, label %bb8608.i, label %bb8396.i
bb8396.i:		; preds = %bb8352.i
	br label %bb8588.i63
bb8492.i:		; preds = %bb8588.i63
	br label %bb8588.i63
bb8588.i63:		; preds = %bb8492.i, %bb8396.i
	br i1 false, label %bb8599.i, label %bb8492.i
bb8599.i:		; preds = %bb8588.i63
	br label %bb8608.i
bb8608.i:		; preds = %bb8599.i, %bb8352.i, %bb8259.i
	br i1 false, label %bb8706.i, label %bb8655.i
bb8655.i:		; preds = %bb8608.i
	br label %bb8990.i
bb8706.i:		; preds = %bb8608.i
	br label %bb8990.i
bb8715.i:		; preds = %bb6924.i, %bb6924.i
	br label %bb8990.i
bb8792.i:		; preds = %bb6924.i
	br label %bb8990.i
bb8929.i:		; preds = %bb6924.i
	br label %bb8990.i
bb8990.i:		; preds = %bb8929.i, %bb8792.i, %bb8715.i, %bb8706.i, %bb8655.i, %bb6924.i, %bb6908.i, %bb6873.i, %bb6853.i48, %bb6844.i, %bb6793.i, %bb6403.i, %bb6332.i, %bb6323.i, %bb6272.i, %bb5882.i, %bb5558.i, %bb5556.i
	switch i32 %sf4083.0.i, label %bb11184.i [
		 i32 0, label %bb10372.i
		 i32 1, label %bb10609.i
		 i32 2, label %bb10811.i
		 i32 3, label %bb11013.i
		 i32 4, label %bb8992.i
		 i32 5, label %bb8992.i
		 i32 6, label %bb8992.i
		 i32 7, label %bb8992.i
		 i32 8, label %bb9195.i
		 i32 9, label %bb9195.i
		 i32 10, label %bb9965.i
		 i32 11, label %bb9585.i
		 i32 16, label %bb9195.i
	]
bb8992.i:		; preds = %bb8990.i, %bb8990.i, %bb8990.i, %bb8990.i
	switch i32 0, label %bb11184.i [
		 i32 0, label %bb9075.i
		 i32 1, label %bb9105.i
		 i32 2, label %bb9135.i
		 i32 3, label %bb9165.i
		 i32 11, label %bb8994.i
	]
bb8994.i:		; preds = %bb8992.i
	br label %bb11247.i
bb9075.i:		; preds = %bb8992.i
	br label %bb11247.i
bb9105.i:		; preds = %bb8992.i
	br label %bb11247.i
bb9135.i:		; preds = %bb8992.i
	br label %bb11247.i
bb9165.i:		; preds = %bb8992.i
	br label %bb11247.i
bb9195.i:		; preds = %bb8990.i, %bb8990.i, %bb8990.i
	switch i32 0, label %bb11184.i [
		 i32 0, label %bb9491.i
		 i32 1, label %bb9521.i
		 i32 2, label %bb9551.i
		 i32 3, label %bb9581.i
		 i32 4, label %bb9197.i
		 i32 11, label %bb9342.i
	]
bb9197.i:		; preds = %bb9195.i
	br label %bb11247.i
bb9342.i:		; preds = %bb9195.i
	br label %bb11247.i
bb9491.i:		; preds = %bb9195.i
	br label %bb11247.i
bb9521.i:		; preds = %bb9195.i
	br label %bb11247.i
bb9551.i:		; preds = %bb9195.i
	br label %bb11247.i
bb9581.i:		; preds = %bb9195.i
	br label %bb11247.i
bb9585.i:		; preds = %bb8990.i
	switch i32 0, label %bb11184.i [
		 i32 0, label %bb9879.i
		 i32 1, label %bb9920.i
		 i32 2, label %bb9920.i
		 i32 3, label %bb9924.i
		 i32 4, label %bb9587.i
		 i32 8, label %bb9587.i
	]
bb9587.i:		; preds = %bb9585.i, %bb9585.i
	br label %bb11247.i
bb9879.i:		; preds = %bb9585.i
	br label %bb11247.i
bb9920.i:		; preds = %bb9585.i, %bb9585.i
	br label %bb11247.i
bb9924.i:		; preds = %bb9585.i
	br label %bb11247.i
bb9965.i:		; preds = %bb8990.i
	switch i32 0, label %bb11184.i [
		 i32 1, label %bb10368.i
		 i32 2, label %bb10368.i
		 i32 3, label %bb10364.i
		 i32 4, label %bb9967.i
		 i32 8, label %bb10127.i
		 i32 11, label %bb10287.i
	]
bb9967.i:		; preds = %bb9965.i
	br label %bb11247.i
bb10127.i:		; preds = %bb9965.i
	br label %bb11247.i
bb10287.i:		; preds = %bb9965.i
	br label %bb11247.i
bb10364.i:		; preds = %bb9965.i
	br label %bb11247.i
bb10368.i:		; preds = %bb9965.i, %bb9965.i
	br label %bb11247.i
bb10372.i:		; preds = %bb8990.i
	switch i32 0, label %bb11184.i [
		 i32 1, label %bb10605.i
		 i32 2, label %bb10605.i
		 i32 3, label %bb10601.i
		 i32 4, label %bb10374.i
		 i32 8, label %bb10449.i
		 i32 11, label %bb10524.i
	]
bb10374.i:		; preds = %bb10372.i
	br label %bb11247.i
bb10449.i:		; preds = %bb10372.i
	br label %bb11247.i
bb10524.i:		; preds = %bb10372.i
	br label %bb11247.i
bb10601.i:		; preds = %bb10372.i
	br label %bb11247.i
bb10605.i:		; preds = %bb10372.i, %bb10372.i
	br label %bb11247.i
bb10609.i:		; preds = %bb8990.i
	switch i32 0, label %bb11184.i [
		 i32 0, label %bb10807.i
		 i32 2, label %bb10807.i
		 i32 3, label %bb10803.i
		 i32 4, label %bb10611.i
		 i32 8, label %bb10686.i
		 i32 11, label %bb10761.i
	]
bb10611.i:		; preds = %bb10609.i
	br label %bb11247.i
bb10686.i:		; preds = %bb10609.i
	br label %bb11247.i
bb10761.i:		; preds = %bb10609.i
	br label %bb11247.i
bb10803.i:		; preds = %bb10609.i
	br label %bb11247.i
bb10807.i:		; preds = %bb10609.i, %bb10609.i
	br label %bb11247.i
bb10811.i:		; preds = %bb8990.i
	switch i32 0, label %bb11184.i [
		 i32 0, label %bb11009.i
		 i32 1, label %bb11009.i
		 i32 3, label %bb11005.i
		 i32 4, label %bb10813.i
		 i32 8, label %bb10888.i
		 i32 11, label %bb10963.i
	]
bb10813.i:		; preds = %bb10811.i
	br label %bb11247.i
bb10888.i:		; preds = %bb10811.i
	br label %bb11247.i
bb10963.i:		; preds = %bb10811.i
	br label %bb11247.i
bb11005.i:		; preds = %bb10811.i
	br label %bb11247.i
bb11009.i:		; preds = %bb10811.i, %bb10811.i
	br label %bb11247.i
bb11013.i:		; preds = %bb8990.i
	switch i32 0, label %bb11184.i [
		 i32 0, label %bb11180.i
		 i32 1, label %bb11180.i
		 i32 2, label %bb11180.i
		 i32 4, label %bb11015.i
		 i32 8, label %bb11090.i
		 i32 11, label %bb11103.i
	]
bb11015.i:		; preds = %bb11013.i
	br label %bb11247.i
bb11090.i:		; preds = %bb11013.i
	br label %bb11247.i
bb11103.i:		; preds = %bb11013.i
	br label %bb11247.i
bb11180.i:		; preds = %bb11013.i, %bb11013.i, %bb11013.i
	br label %bb11184.i
bb11184.i:		; preds = %bb11180.i, %bb11013.i, %bb10811.i, %bb10609.i, %bb10372.i, %bb9965.i, %bb9585.i, %bb9195.i, %bb8992.i, %bb8990.i
	br label %bb11247.i
bb11247.i:		; preds = %bb11184.i, %bb11103.i, %bb11090.i, %bb11015.i, %bb11009.i, %bb11005.i, %bb10963.i, %bb10888.i, %bb10813.i, %bb10807.i, %bb10803.i, %bb10761.i, %bb10686.i, %bb10611.i, %bb10605.i, %bb10601.i, %bb10524.i, %bb10449.i, %bb10374.i, %bb10368.i, %bb10364.i, %bb10287.i, %bb10127.i, %bb9967.i, %bb9924.i, %bb9920.i, %bb9879.i, %bb9587.i, %bb9581.i, %bb9551.i, %bb9521.i, %bb9491.i, %bb9342.i, %bb9197.i, %bb9165.i, %bb9135.i, %bb9105.i, %bb9075.i, %bb8994.i
	br i1 false, label %bb11250.i, label %bb11256.i
bb11250.i:		; preds = %bb11247.i
	br label %bb11378.i
bb11256.i:		; preds = %bb11247.i
	switch i32 0, label %bb11348.i [
		 i32 4, label %bb11258.i
		 i32 8, label %bb11258.i
		 i32 11, label %bb11318.i
	]
bb11258.i:		; preds = %bb11256.i, %bb11256.i
	br i1 false, label %bb11273.i, label %bb11261.i
bb11261.i:		; preds = %bb11258.i
	br label %bb11273.i
bb11273.i:		; preds = %bb11261.i, %bb11258.i
	br i1 false, label %bb11288.i, label %bb11276.i
bb11276.i:		; preds = %bb11273.i
	br label %bb11288.i
bb11288.i:		; preds = %bb11276.i, %bb11273.i
	br i1 false, label %bb11303.i, label %bb11291.i
bb11291.i:		; preds = %bb11288.i
	br label %bb11303.i
bb11303.i:		; preds = %bb11291.i, %bb11288.i
	br i1 false, label %bb11318.i, label %bb11306.i
bb11306.i:		; preds = %bb11303.i
	br label %bb11318.i
bb11318.i:		; preds = %bb11306.i, %bb11303.i, %bb11256.i
	br i1 false, label %bb11333.i, label %bb11321.i
bb11321.i:		; preds = %bb11318.i
	br label %bb11333.i
bb11333.i:		; preds = %bb11321.i, %bb11318.i
	br i1 false, label %bb11348.i, label %bb11336.i
bb11336.i:		; preds = %bb11333.i
	br label %bb11348.i
bb11348.i:		; preds = %bb11336.i, %bb11333.i, %bb11256.i
	br i1 false, label %bb11363.i, label %bb11351.i
bb11351.i:		; preds = %bb11348.i
	br label %bb11363.i
bb11363.i:		; preds = %bb11351.i, %bb11348.i
	br i1 false, label %bb11378.i, label %bb11366.i
bb11366.i:		; preds = %bb11363.i
	br label %bb11378.i
bb11378.i:		; preds = %bb11366.i, %bb11363.i, %bb11250.i
	br label %bb12038.i
bb12038.i:		; preds = %bb11378.i, %bb5464.i, %bb5404.i, %bb5374.i, %bb5344.i, %bb5263.i
	switch i32 0, label %bb15866.i [
		 i32 3, label %bb13016.i
		 i32 4, label %bb12040.i
		 i32 8, label %bb12514.i
		 i32 10, label %bb12903.i
		 i32 11, label %bb12553.i
		 i32 16, label %bb12514.i
	]
bb12040.i:		; preds = %bb12038.i, %bb5467.i
	br label %bb13026.i
bb12514.i:		; preds = %bb12038.i, %bb12038.i, %bb5467.i, %bb5467.i
	br label %bb13026.i
bb12553.i:		; preds = %bb12038.i, %bb5467.i
	br i1 false, label %bb12558.i, label %bb12747.i
bb12558.i:		; preds = %bb12553.i
	br i1 false, label %bb12666.i, label %bb12654.i
bb12654.i:		; preds = %bb12558.i
	br label %bb12666.i
bb12666.i:		; preds = %bb12654.i, %bb12558.i
	br label %bb12747.i
bb12747.i:		; preds = %bb12666.i, %bb12553.i
	br label %bb13026.i
bb12903.i:		; preds = %bb12038.i, %bb5467.i
	br i1 false, label %bb12908.i, label %bb13026.i
bb12908.i:		; preds = %bb12903.i
	br i1 false, label %bb13026.i, label %bb13004.i
bb13004.i:		; preds = %bb12908.i
	switch i32 0, label %bb15866.i [
		 i32 3, label %bb13752.i
		 i32 4, label %bb14197.i
		 i32 8, label %bb14197.i
		 i32 10, label %bb13752.i
		 i32 11, label %bb13307.i
		 i32 16, label %bb13028.i
	]
bb13016.i:		; preds = %bb12038.i, %bb5467.i
	br label %bb13026.i
bb13026.i:		; preds = %bb13016.i, %bb12908.i, %bb12903.i, %bb12747.i, %bb12514.i, %bb12040.i
	switch i32 0, label %bb15866.i [
		 i32 3, label %bb13752.i
		 i32 4, label %bb14197.i
		 i32 8, label %bb14197.i
		 i32 10, label %bb13752.i
		 i32 11, label %bb13307.i
		 i32 16, label %bb13028.i
	]
bb13028.i:		; preds = %bb13026.i, %bb13004.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb13307.i:		; preds = %bb13026.i, %bb13004.i
	switch i32 %dt4080.0.i, label %bb13736.i [
		 i32 6, label %bb13312.i
		 i32 1, label %bb13624.i
		 i32 3, label %bb13624.i
		 i32 5, label %bb13649.i
		 i32 4, label %bb13688.i
		 i32 7, label %bb15866.i
	]
bb13312.i:		; preds = %bb13307.i
	br i1 false, label %bb13483.i, label %bb13400.i
bb13400.i:		; preds = %bb13312.i
	br label %bb13483.i
bb13483.i:		; preds = %bb13400.i, %bb13312.i
	br i1 false, label %bb13593.i, label %bb13505.i
bb13505.i:		; preds = %bb13483.i
	switch i32 %dt4080.0.i, label %bb14181.i [
		 i32 6, label %bb13757.i
		 i32 1, label %bb14069.i
		 i32 3, label %bb14069.i
		 i32 5, label %bb14094.i
		 i32 4, label %bb14133.i
		 i32 7, label %bb15866.i
	]
bb13593.i:		; preds = %bb13483.i
	switch i32 %dt4080.0.i, label %bb14181.i [
		 i32 6, label %bb13757.i
		 i32 1, label %bb14069.i
		 i32 3, label %bb14069.i
		 i32 5, label %bb14094.i
		 i32 4, label %bb14133.i
		 i32 7, label %bb15866.i
	]
bb13624.i:		; preds = %bb13307.i, %bb13307.i
	switch i32 %dt4080.0.i, label %bb14181.i [
		 i32 6, label %bb13757.i
		 i32 1, label %bb14069.i
		 i32 3, label %bb14069.i
		 i32 5, label %bb14094.i
		 i32 4, label %bb14133.i
		 i32 7, label %bb15866.i
	]
bb13649.i:		; preds = %bb13307.i
	br label %bb14094.i
bb13688.i:		; preds = %bb13307.i
	br label %bb14133.i
bb13736.i:		; preds = %bb13307.i
	br label %bb13752.i
bb13752.i:		; preds = %bb13736.i, %bb13026.i, %bb13026.i, %bb13004.i, %bb13004.i
	switch i32 %dt4080.0.i, label %bb14181.i [
		 i32 6, label %bb13757.i
		 i32 1, label %bb14069.i
		 i32 3, label %bb14069.i
		 i32 5, label %bb14094.i
		 i32 4, label %bb14133.i
		 i32 7, label %bb15866.i
	]
bb13757.i:		; preds = %bb13752.i, %bb13624.i, %bb13593.i, %bb13505.i
	br i1 false, label %bb13928.i, label %bb13845.i
bb13845.i:		; preds = %bb13757.i
	br label %bb13928.i
bb13928.i:		; preds = %bb13845.i, %bb13757.i
	br i1 false, label %bb14038.i, label %bb13950.i
bb13950.i:		; preds = %bb13928.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb14038.i:		; preds = %bb13928.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb14069.i:		; preds = %bb13752.i, %bb13752.i, %bb13624.i, %bb13624.i, %bb13593.i, %bb13593.i, %bb13505.i, %bb13505.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb14094.i:		; preds = %bb13752.i, %bb13649.i, %bb13624.i, %bb13593.i, %bb13505.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb14133.i:		; preds = %bb13752.i, %bb13688.i, %bb13624.i, %bb13593.i, %bb13505.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb14181.i:		; preds = %bb13752.i, %bb13624.i, %bb13593.i, %bb13505.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb14197.i:		; preds = %bb13026.i, %bb13026.i, %bb13004.i, %bb13004.i
	switch i32 %dt4080.0.i, label %bb15805.i [
		 i32 6, label %bb14202.i
		 i32 1, label %bb15411.i
		 i32 3, label %bb15411.i
		 i32 5, label %bb15493.i
		 i32 4, label %bb15631.i
		 i32 7, label %bb15866.i
	]
bb14202.i:		; preds = %bb14197.i
	br i1 false, label %bb14373.i, label %bb14290.i
bb14290.i:		; preds = %bb14202.i
	br label %bb14373.i
bb14373.i:		; preds = %bb14290.i, %bb14202.i
	br i1 false, label %bb14483.i, label %bb14395.i
bb14395.i:		; preds = %bb14373.i
	br label %bb14483.i
bb14483.i:		; preds = %bb14395.i, %bb14373.i
	br i1 false, label %bb14672.i, label %bb14589.i
bb14589.i:		; preds = %bb14483.i
	br label %bb14672.i
bb14672.i:		; preds = %bb14589.i, %bb14483.i
	br i1 false, label %bb14782.i, label %bb14694.i
bb14694.i:		; preds = %bb14672.i
	br label %bb14782.i
bb14782.i:		; preds = %bb14694.i, %bb14672.i
	br i1 false, label %bb14971.i, label %bb14888.i
bb14888.i:		; preds = %bb14782.i
	br label %bb14971.i
bb14971.i:		; preds = %bb14888.i, %bb14782.i
	br i1 false, label %bb15081.i, label %bb14993.i
bb14993.i:		; preds = %bb14971.i
	br label %bb15081.i
bb15081.i:		; preds = %bb14993.i, %bb14971.i
	br i1 false, label %bb15270.i, label %bb15187.i
bb15187.i:		; preds = %bb15081.i
	br label %bb15270.i
bb15270.i:		; preds = %bb15187.i, %bb15081.i
	br i1 false, label %bb15380.i, label %bb15292.i
bb15292.i:		; preds = %bb15270.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb15380.i:		; preds = %bb15270.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb15411.i:		; preds = %bb14197.i, %bb14197.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb15493.i:		; preds = %bb14197.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb15631.i:		; preds = %bb14197.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb15805.i:		; preds = %bb14197.i
	br label %bb15866.i
bb15866.i:		; preds = %bb15805.i, %bb14197.i, %bb13752.i, %bb13624.i, %bb13593.i, %bb13505.i, %bb13307.i, %bb13026.i, %bb13004.i, %bb12038.i, %bb5467.i
	br i1 false, label %UnifiedReturnBlock.i177, label %bb15869.i
bb15869.i:		; preds = %bb15866.i, %bb15631.i, %bb15493.i, %bb15411.i, %bb15380.i, %bb15292.i, %bb14181.i, %bb14133.i, %bb14094.i, %bb14069.i, %bb14038.i, %bb13950.i, %bb13028.i
	switch i32 0, label %UnifiedReturnBlock.i177 [
		 i32 4, label %bb15874.i
		 i32 8, label %bb15960.i
	]
bb15874.i:		; preds = %bb15869.i
	br label %glgVectorFloatConversion.exit
bb15960.i:		; preds = %bb15869.i
	br label %glgVectorFloatConversion.exit
UnifiedReturnBlock.i177:		; preds = %bb15869.i, %bb15866.i, %bb15631.i, %bb15493.i, %bb15411.i, %bb15380.i, %bb15292.i, %bb14181.i, %bb14133.i, %bb14094.i, %bb14069.i, %bb14038.i, %bb13950.i, %bb13028.i
	br label %glgVectorFloatConversion.exit
glgVectorFloatConversion.exit:		; preds = %UnifiedReturnBlock.i177, %bb15960.i, %bb15874.i
	br label %bb16581.i
bb5351.i:		; preds = %loadVecColor_BGRA_UI8888R.exit
	br i1 false, label %bb5359.i, label %bb5586.i
bb5359.i:		; preds = %bb5351.i
	switch i32 0, label %bb5586.i [
		 i32 0, label %bb5361.i
		 i32 1, label %bb5511.i
		 i32 2, label %bb5511.i
	]
bb5361.i:		; preds = %bb5359.i
	br i1 false, label %bb5366.i, label %bb5379.i
bb5366.i:		; preds = %bb5361.i
	br label %bb7230.i
bb5379.i:		; preds = %bb5361.i
	switch i32 %sf4083.0.i, label %bb5415.i [
		 i32 1, label %bb5384.i
		 i32 2, label %bb5402.i
	]
bb5384.i:		; preds = %bb5379.i
	switch i32 0, label %bb7230.i [
		 i32 4, label %bb5445.i
		 i32 8, label %bb5445.i
		 i32 11, label %bb5445.i
	]
bb5402.i:		; preds = %bb5379.i
	switch i32 0, label %bb7230.i [
		 i32 4, label %bb5445.i
		 i32 8, label %bb5445.i
		 i32 11, label %bb5445.i
	]
bb5415.i:		; preds = %bb5379.i
	switch i32 0, label %bb7230.i [
		 i32 4, label %bb5445.i
		 i32 8, label %bb5445.i
		 i32 11, label %bb5445.i
	]
bb5445.i:		; preds = %bb5415.i, %bb5415.i, %bb5415.i, %bb5402.i, %bb5402.i, %bb5402.i, %bb5384.i, %bb5384.i, %bb5384.i
	switch i32 0, label %bb7230.i [
		 i32 4, label %bb5470.i
		 i32 8, label %bb5470.i
		 i32 11, label %bb6853.i
	]
bb5470.i:		; preds = %bb5445.i, %bb5445.i
	switch i32 0, label %bb7230.i [
		 i32 4, label %bb5498.i
		 i32 8, label %bb5493.i
		 i32 11, label %bb6853.i
	]
bb5493.i:		; preds = %bb5470.i
	br i1 false, label %bb5498.i, label %bb5586.i
bb5498.i:		; preds = %bb5493.i, %bb5470.i
	switch i32 0, label %bb7230.i [
		 i32 4, label %bb5591.i
		 i32 8, label %bb6153.i
		 i32 11, label %bb6853.i
	]
bb5511.i:		; preds = %bb5359.i, %bb5359.i
	br i1 false, label %bb5568.i, label %bb5586.i
bb5568.i:		; preds = %bb5511.i
	br label %bb5586.i
bb5586.i:		; preds = %bb5568.i, %bb5511.i, %bb5493.i, %bb5359.i, %bb5351.i
	switch i32 0, label %bb7230.i [
		 i32 4, label %bb5591.i
		 i32 8, label %bb6153.i
		 i32 11, label %bb6853.i
	]
bb5591.i:		; preds = %bb5586.i, %bb5498.i
	switch i32 0, label %bb5995.i [
		 i32 4, label %bb5596.i
		 i32 8, label %bb5680.i
		 i32 11, label %bb5842.i
	]
bb5596.i:		; preds = %bb5591.i
	br i1 false, label %bb8428.i, label %bb5602.i
bb5602.i:		; preds = %bb5596.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb5680.i:		; preds = %bb5591.i
	br i1 false, label %bb5692.i, label %bb5764.i
bb5692.i:		; preds = %bb5680.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb5764.i:		; preds = %bb5680.i
	br i1 false, label %bb8428.i, label %bb5772.i
bb5772.i:		; preds = %bb5764.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb5842.i:		; preds = %bb5591.i
	br i1 false, label %bb5920.i, label %bb5845.i
bb5845.i:		; preds = %bb5842.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb5920.i:		; preds = %bb5842.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb5995.i:		; preds = %bb5591.i
	switch i32 %df4081.0.i, label %bb8428.i [
		 i32 0, label %bb6007.i
		 i32 10, label %bb6007.i
		 i32 1, label %bb6042.i
		 i32 2, label %bb6079.i
		 i32 3, label %bb6116.i
	]
bb6007.i:		; preds = %bb5995.i, %bb5995.i
	br i1 false, label %bb6012.i, label %bb8428.i
bb6012.i:		; preds = %bb6007.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6042.i:		; preds = %bb5995.i
	br i1 false, label %bb6049.i, label %bb6045.i
bb6045.i:		; preds = %bb6042.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6049.i:		; preds = %bb6042.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6079.i:		; preds = %bb5995.i
	br i1 false, label %bb6086.i, label %bb6082.i
bb6082.i:		; preds = %bb6079.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6086.i:		; preds = %bb6079.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6116.i:		; preds = %bb5995.i
	br i1 false, label %bb6123.i, label %bb6119.i
bb6119.i:		; preds = %bb6116.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6123.i:		; preds = %bb6116.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6153.i:		; preds = %bb5586.i, %bb5498.i
	switch i32 0, label %bb6724.i [
		 i32 4, label %bb6158.i
		 i32 8, label %bb6459.i
		 i32 11, label %bb6621.i
	]
bb6158.i:		; preds = %bb6153.i
	br i1 false, label %bb6242.i, label %bb6161.i
bb6161.i:		; preds = %bb6158.i
	br i1 false, label %bb6239.i, label %bb6166.i
bb6166.i:		; preds = %bb6161.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6239.i:		; preds = %bb6161.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6242.i:		; preds = %bb6158.i
	br i1 false, label %bb6245.i, label %bb6317.i
bb6245.i:		; preds = %bb6242.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6317.i:		; preds = %bb6242.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6459.i:		; preds = %bb6153.i
	br i1 false, label %bb6471.i, label %bb6543.i
bb6471.i:		; preds = %bb6459.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6543.i:		; preds = %bb6459.i
	br i1 false, label %bb8428.i, label %bb6551.i
bb6551.i:		; preds = %bb6543.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6621.i:		; preds = %bb6153.i
	br i1 false, label %bb6626.i, label %bb6651.i
bb6626.i:		; preds = %bb6621.i
	br label %bb6651.i
bb6651.i:		; preds = %bb6626.i, %bb6621.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6724.i:		; preds = %bb6153.i
	switch i32 %df4081.0.i, label %bb8428.i [
		 i32 0, label %bb6736.i
		 i32 10, label %bb6736.i
		 i32 1, label %bb6771.i
		 i32 2, label %bb6808.i
		 i32 3, label %bb6845.i
	]
bb6736.i:		; preds = %bb6724.i, %bb6724.i
	br i1 false, label %bb6741.i, label %bb8428.i
bb6741.i:		; preds = %bb6736.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6771.i:		; preds = %bb6724.i
	br i1 false, label %bb6778.i, label %bb6774.i
bb6774.i:		; preds = %bb6771.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6778.i:		; preds = %bb6771.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6808.i:		; preds = %bb6724.i
	br i1 false, label %bb6815.i, label %bb6811.i
bb6811.i:		; preds = %bb6808.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6815.i:		; preds = %bb6808.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6845.i:		; preds = %bb6724.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6853.i:		; preds = %bb5586.i, %bb5498.i, %bb5470.i, %bb5445.i
	switch i32 0, label %bb8428.i [
		 i32 4, label %bb6858.i
		 i32 8, label %bb7072.i
		 i32 10, label %bb7149.i
		 i32 3, label %bb7192.i
	]
bb6858.i:		; preds = %bb6853.i
	br i1 false, label %bb6942.i, label %bb6861.i
bb6861.i:		; preds = %bb6858.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb6942.i:		; preds = %bb6858.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7072.i:		; preds = %bb6853.i
	br i1 false, label %bb7119.i, label %bb7075.i
bb7075.i:		; preds = %bb7072.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7119.i:		; preds = %bb7072.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7149.i:		; preds = %bb6853.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7192.i:		; preds = %bb6853.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7230.i:		; preds = %bb5586.i, %bb5498.i, %bb5470.i, %bb5445.i, %bb5415.i, %bb5402.i, %bb5384.i, %bb5366.i
	switch i32 %sf4083.0.i, label %bb8428.i [
		 i32 10, label %bb7235.i
		 i32 0, label %bb7455.i
		 i32 1, label %bb7725.i
		 i32 2, label %bb7978.i
		 i32 3, label %bb8231.i
	]
bb7235.i:		; preds = %bb7230.i
	switch i32 0, label %bb7442.i [
		 i32 4, label %bb7240.i
		 i32 8, label %bb7329.i
		 i32 11, label %bb7369.i
	]
bb7240.i:		; preds = %bb7235.i
	br i1 false, label %bb7252.i, label %bb7243.i
bb7243.i:		; preds = %bb7240.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7252.i:		; preds = %bb7240.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7329.i:		; preds = %bb7235.i
	br i1 false, label %bb7339.i, label %bb7332.i
bb7332.i:		; preds = %bb7329.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7339.i:		; preds = %bb7329.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7369.i:		; preds = %bb7235.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7442.i:		; preds = %bb7235.i
	br i1 false, label %bb7447.i, label %bb8428.i
bb7447.i:		; preds = %bb7442.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7455.i:		; preds = %bb7230.i
	switch i32 0, label %bb7703.i [
		 i32 4, label %bb7460.i
		 i32 8, label %bb7546.i
		 i32 11, label %bb7630.i
	]
bb7460.i:		; preds = %bb7455.i
	br i1 false, label %bb7471.i, label %bb7463.i
bb7463.i:		; preds = %bb7460.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7471.i:		; preds = %bb7460.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7546.i:		; preds = %bb7455.i
	br i1 false, label %bb7555.i, label %bb7549.i
bb7549.i:		; preds = %bb7546.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7555.i:		; preds = %bb7546.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7630.i:		; preds = %bb7455.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7703.i:		; preds = %bb7455.i
	br i1 false, label %bb7709.i, label %bb7712.i
bb7709.i:		; preds = %bb7703.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7712.i:		; preds = %bb7703.i
	br i1 false, label %bb7717.i, label %bb8428.i
bb7717.i:		; preds = %bb7712.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7725.i:		; preds = %bb7230.i
	switch i32 0, label %bb7945.i [
		 i32 4, label %bb7730.i
		 i32 8, label %bb7819.i
		 i32 11, label %bb7906.i
	]
bb7730.i:		; preds = %bb7725.i
	br i1 false, label %bb7744.i, label %bb7733.i
bb7733.i:		; preds = %bb7730.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7744.i:		; preds = %bb7730.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7819.i:		; preds = %bb7725.i
	br i1 false, label %bb7831.i, label %bb7822.i
bb7822.i:		; preds = %bb7819.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7831.i:		; preds = %bb7819.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7906.i:		; preds = %bb7725.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7945.i:		; preds = %bb7725.i
	switch i32 %df4081.0.i, label %bb8428.i [
		 i32 0, label %bb7962.i
		 i32 2, label %bb7962.i
		 i32 10, label %bb7962.i
		 i32 3, label %bb7970.i
	]
bb7962.i:		; preds = %bb7945.i, %bb7945.i, %bb7945.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7970.i:		; preds = %bb7945.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7978.i:		; preds = %bb7230.i
	switch i32 0, label %bb8198.i [
		 i32 4, label %bb7983.i
		 i32 8, label %bb8072.i
		 i32 11, label %bb8159.i
	]
bb7983.i:		; preds = %bb7978.i
	br i1 false, label %bb7997.i, label %bb7986.i
bb7986.i:		; preds = %bb7983.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb7997.i:		; preds = %bb7983.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8072.i:		; preds = %bb7978.i
	br i1 false, label %bb8084.i, label %bb8075.i
bb8075.i:		; preds = %bb8072.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8084.i:		; preds = %bb8072.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8159.i:		; preds = %bb7978.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8198.i:		; preds = %bb7978.i
	switch i32 %df4081.0.i, label %bb8428.i [
		 i32 0, label %bb8215.i
		 i32 1, label %bb8215.i
		 i32 10, label %bb8215.i
		 i32 3, label %bb8223.i
	]
bb8215.i:		; preds = %bb8198.i, %bb8198.i, %bb8198.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8223.i:		; preds = %bb8198.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8231.i:		; preds = %bb7230.i
	switch i32 0, label %bb8428.i [
		 i32 4, label %bb8236.i
		 i32 8, label %bb8326.i
		 i32 11, label %bb8347.i
		 i32 10, label %bb8425.i
	]
bb8236.i:		; preds = %bb8231.i
	br i1 false, label %bb8251.i, label %bb8239.i
bb8239.i:		; preds = %bb8236.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8251.i:		; preds = %bb8236.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8326.i:		; preds = %bb8231.i
	br i1 false, label %bb8339.i, label %bb8428.i
bb8339.i:		; preds = %bb8326.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8347.i:		; preds = %bb8231.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8425.i:		; preds = %bb8231.i
	br label %bb8428.i
bb8428.i:		; preds = %bb8425.i, %bb8326.i, %bb8231.i, %bb8198.i, %bb7945.i, %bb7712.i, %bb7442.i, %bb7230.i, %bb6853.i, %bb6736.i, %bb6724.i, %bb6543.i, %bb6007.i, %bb5995.i, %bb5764.i, %bb5596.i
	br i1 false, label %bb8668.i, label %bb8434.i
bb8434.i:		; preds = %bb8428.i, %bb8347.i, %bb8339.i, %bb8251.i, %bb8239.i, %bb8223.i, %bb8215.i, %bb8159.i, %bb8084.i, %bb8075.i, %bb7997.i, %bb7986.i, %bb7970.i, %bb7962.i, %bb7906.i, %bb7831.i, %bb7822.i, %bb7744.i, %bb7733.i, %bb7717.i, %bb7709.i, %bb7630.i, %bb7555.i, %bb7549.i, %bb7471.i, %bb7463.i, %bb7447.i, %bb7369.i, %bb7339.i, %bb7332.i, %bb7252.i, %bb7243.i, %bb7192.i, %bb7149.i, %bb7119.i, %bb7075.i, %bb6942.i, %bb6861.i, %bb6845.i, %bb6815.i, %bb6811.i, %bb6778.i, %bb6774.i, %bb6741.i, %bb6651.i, %bb6551.i, %bb6471.i, %bb6317.i, %bb6245.i, %bb6239.i, %bb6166.i, %bb6123.i, %bb6119.i, %bb6086.i, %bb6082.i, %bb6049.i, %bb6045.i, %bb6012.i, %bb5920.i, %bb5845.i, %bb5772.i, %bb5692.i, %bb5602.i
	switch i32 0, label %bb8668.i [
		 i32 0, label %bb8436.i
		 i32 1, label %bb8531.i
		 i32 2, label %bb8531.i
	]
bb8436.i:		; preds = %bb8434.i
	switch i32 0, label %bb9310.i [
		 i32 4, label %bb8465.i
		 i32 8, label %bb8465.i
		 i32 11, label %bb8465.i
		 i32 3, label %bb9301.i
	]
bb8465.i:		; preds = %bb8436.i, %bb8436.i, %bb8436.i
	switch i32 0, label %bb9310.i [
		 i32 4, label %bb8490.i
		 i32 8, label %bb8490.i
		 i32 3, label %bb9301.i
		 i32 11, label %bb9153.i
	]
bb8490.i:		; preds = %bb8465.i, %bb8465.i
	switch i32 0, label %bb9310.i [
		 i32 4, label %bb8518.i
		 i32 8, label %bb8513.i
		 i32 3, label %bb9301.i
		 i32 11, label %bb9153.i
	]
bb8513.i:		; preds = %bb8490.i
	br i1 false, label %bb8518.i, label %bb8668.i
bb8518.i:		; preds = %bb8513.i, %bb8490.i
	switch i32 0, label %bb9310.i [
		 i32 3, label %bb9301.i
		 i32 4, label %bb8670.i
		 i32 8, label %bb9112.i
		 i32 11, label %bb9153.i
	]
bb8531.i:		; preds = %bb8434.i, %bb8434.i
	br i1 false, label %bb8536.i, label %bb8575.i
bb8536.i:		; preds = %bb8531.i
	br i1 false, label %bb8557.i, label %bb8588.i
bb8557.i:		; preds = %bb8536.i
	switch i32 0, label %bb9310.i [
		 i32 4, label %bb8600.i
		 i32 8, label %bb8600.i
		 i32 3, label %bb9301.i
		 i32 11, label %bb9153.i
	]
bb8575.i:		; preds = %bb8531.i
	br label %bb8588.i
bb8588.i:		; preds = %bb8575.i, %bb8536.i
	switch i32 0, label %bb9310.i [
		 i32 4, label %bb8600.i
		 i32 8, label %bb8600.i
		 i32 3, label %bb9301.i
		 i32 11, label %bb9153.i
	]
bb8600.i:		; preds = %bb8588.i, %bb8588.i, %bb8557.i, %bb8557.i
	switch i32 0, label %bb9310.i [
		 i32 4, label %bb8629.i
		 i32 3, label %bb9301.i
		 i32 8, label %bb9112.i
		 i32 11, label %bb9153.i
	]
bb8629.i:		; preds = %bb8600.i
	br i1 false, label %bb8650.i, label %bb8668.i
bb8650.i:		; preds = %bb8629.i
	br label %bb8668.i
bb8668.i:		; preds = %bb8650.i, %bb8629.i, %bb8513.i, %bb8434.i, %bb8428.i, %bb8347.i, %bb8339.i, %bb8251.i, %bb8239.i, %bb8223.i, %bb8215.i, %bb8159.i, %bb8084.i, %bb8075.i, %bb7997.i, %bb7986.i, %bb7970.i, %bb7962.i, %bb7906.i, %bb7831.i, %bb7822.i, %bb7744.i, %bb7733.i, %bb7717.i, %bb7709.i, %bb7630.i, %bb7555.i, %bb7549.i, %bb7471.i, %bb7463.i, %bb7447.i, %bb7369.i, %bb7339.i, %bb7332.i, %bb7252.i, %bb7243.i, %bb7192.i, %bb7149.i, %bb7119.i, %bb7075.i, %bb6942.i, %bb6861.i, %bb6845.i, %bb6815.i, %bb6811.i, %bb6778.i, %bb6774.i, %bb6741.i, %bb6651.i, %bb6551.i, %bb6471.i, %bb6317.i, %bb6245.i, %bb6239.i, %bb6166.i, %bb6123.i, %bb6119.i, %bb6086.i, %bb6082.i, %bb6049.i, %bb6045.i, %bb6012.i, %bb5920.i, %bb5845.i, %bb5772.i, %bb5692.i, %bb5602.i
	switch i32 0, label %bb9310.i [
		 i32 3, label %bb9301.i
		 i32 4, label %bb8670.i
		 i32 8, label %bb9112.i
		 i32 11, label %bb9153.i
	]
bb8670.i:		; preds = %bb8668.i, %bb8518.i
	br label %bb9310.i
bb9112.i:		; preds = %bb8668.i, %bb8600.i, %bb8518.i
	br label %bb9310.i
bb9153.i:		; preds = %bb8668.i, %bb8600.i, %bb8588.i, %bb8557.i, %bb8518.i, %bb8490.i, %bb8465.i
	br label %bb9310.i
bb9301.i:		; preds = %bb8668.i, %bb8600.i, %bb8588.i, %bb8557.i, %bb8518.i, %bb8490.i, %bb8465.i, %bb8436.i
	br label %bb9310.i
bb9310.i:		; preds = %bb9301.i, %bb9153.i, %bb9112.i, %bb8670.i, %bb8668.i, %bb8600.i, %bb8588.i, %bb8557.i, %bb8518.i, %bb8490.i, %bb8465.i, %bb8436.i
	br i1 false, label %bb16581.i, label %bb9313.i
bb9313.i:		; preds = %bb9310.i
	switch i32 %dt4080.0.i, label %bb16578.i [
		 i32 0, label %bb9315.i
		 i32 1, label %bb9890.i
		 i32 2, label %bb10465.i
		 i32 3, label %bb11040.i
		 i32 4, label %bb11615.i
		 i32 5, label %bb11823.i
		 i32 8, label %bb12398.i
		 i32 9, label %bb12833.i
		 i32 10, label %bb13268.i
		 i32 11, label %bb13268.i
		 i32 12, label %bb13703.i
		 i32 13, label %bb13703.i
		 i32 14, label %bb14278.i
		 i32 15, label %bb14853.i
		 i32 16, label %bb9315.i
		 i32 17, label %bb9315.i
		 i32 18, label %bb15428.i
		 i32 19, label %bb16003.i
	]
bb9315.i:		; preds = %bb9313.i, %bb9313.i, %bb9313.i
	br i1 false, label %bb9535.i, label %bb9323.i
bb9323.i:		; preds = %bb9315.i
	br label %bb9535.i
bb9535.i:		; preds = %bb9323.i, %bb9315.i
	br label %bb16581.i
bb9890.i:		; preds = %bb9313.i
	br i1 false, label %bb10255.i, label %bb9898.i
bb9898.i:		; preds = %bb9890.i
	br label %bb10255.i
bb10255.i:		; preds = %bb9898.i, %bb9890.i
	br label %bb16581.i
bb10465.i:		; preds = %bb9313.i
	br i1 false, label %bb10685.i, label %bb10473.i
bb10473.i:		; preds = %bb10465.i
	br label %bb10685.i
bb10685.i:		; preds = %bb10473.i, %bb10465.i
	br label %bb16581.i
bb11040.i:		; preds = %bb9313.i
	br i1 false, label %bb11405.i, label %bb11048.i
bb11048.i:		; preds = %bb11040.i
	br label %bb11405.i
bb11405.i:		; preds = %bb11048.i, %bb11040.i
	br label %bb16581.i
bb11615.i:		; preds = %bb9313.i
	br i1 false, label %bb16581.i, label %bb11618.i
bb11618.i:		; preds = %bb11615.i
	br label %bb16581.i
bb11823.i:		; preds = %bb9313.i
	br i1 false, label %bb12188.i, label %bb11831.i
bb11831.i:		; preds = %bb11823.i
	br label %bb12188.i
bb12188.i:		; preds = %bb11831.i, %bb11823.i
	br label %bb16581.i
bb12398.i:		; preds = %bb9313.i
	br i1 false, label %bb12566.i, label %bb12406.i
bb12406.i:		; preds = %bb12398.i
	br label %bb12566.i
bb12566.i:		; preds = %bb12406.i, %bb12398.i
	br label %bb16581.i
bb12833.i:		; preds = %bb9313.i
	br i1 false, label %bb13001.i, label %bb12841.i
bb12841.i:		; preds = %bb12833.i
	br label %bb13001.i
bb13001.i:		; preds = %bb12841.i, %bb12833.i
	br label %bb16581.i
bb13268.i:		; preds = %bb9313.i, %bb9313.i
	br i1 false, label %bb13436.i, label %bb13276.i
bb13276.i:		; preds = %bb13268.i
	br label %bb13436.i
bb13436.i:		; preds = %bb13276.i, %bb13268.i
	br label %bb16581.i
bb13703.i:		; preds = %bb9313.i, %bb9313.i
	br i1 false, label %bb13923.i, label %bb13711.i
bb13711.i:		; preds = %bb13703.i
	br label %bb13923.i
bb13923.i:		; preds = %bb13711.i, %bb13703.i
	br label %bb16581.i
bb14278.i:		; preds = %bb9313.i
	br i1 false, label %bb14498.i, label %bb14286.i
bb14286.i:		; preds = %bb14278.i
	br label %bb14498.i
bb14498.i:		; preds = %bb14286.i, %bb14278.i
	br label %bb16581.i
bb14853.i:		; preds = %bb9313.i
	br i1 false, label %bb15073.i, label %bb14861.i
bb14861.i:		; preds = %bb14853.i
	br label %bb15073.i
bb15073.i:		; preds = %bb14861.i, %bb14853.i
	br label %bb16581.i
bb15428.i:		; preds = %bb9313.i
	br i1 false, label %bb15648.i, label %bb15436.i
bb15436.i:		; preds = %bb15428.i
	br label %bb15648.i
bb15648.i:		; preds = %bb15436.i, %bb15428.i
	br label %bb16581.i
bb16003.i:		; preds = %bb9313.i
	br i1 false, label %bb16223.i, label %bb16011.i
bb16011.i:		; preds = %bb16003.i
	br label %bb16223.i
bb16223.i:		; preds = %bb16011.i, %bb16003.i
	br label %bb16581.i
bb16578.i:		; preds = %bb9313.i
	unreachable
bb16581.i:		; preds = %bb16223.i, %bb15648.i, %bb15073.i, %bb14498.i, %bb13923.i, %bb13436.i, %bb13001.i, %bb12566.i, %bb12188.i, %bb11618.i, %bb11615.i, %bb11405.i, %bb10685.i, %bb10255.i, %bb9535.i, %bb9310.i, %glgVectorFloatConversion.exit
	br label %storeVecColor_RGB_UI.exit
storeVecColor_RGB_UI.exit:		; preds = %bb16581.i
	br i1 false, label %bb5295.i, label %bb16621.i
bb16607.i:		; preds = %bb5276.i
	br i1 false, label %bb5295.preheader.i, label %bb16621.i
bb5295.preheader.i:		; preds = %bb16607.i
	br label %bb5295.i
bb16621.i:		; preds = %bb16607.i, %storeVecColor_RGB_UI.exit
	br label %bb16650.outer.i
bb16650.outer.i:		; preds = %bb16621.i
	br label %bb16650.i
bb16650.i:		; preds = %storeColor_RGB_UI.exit, %bb16650.outer.i
	br label %loadColor_BGRA_UI8888R.exit
loadColor_BGRA_UI8888R.exit:		; preds = %bb16650.i
	br i1 false, label %bb16671.i, label %bb16697.i
bb16671.i:		; preds = %loadColor_BGRA_UI8888R.exit
	br i1 false, label %bb.i179, label %bb662.i
bb.i179:		; preds = %bb16671.i
	switch i32 0, label %bb513.i [
		 i32 7, label %bb418.i
		 i32 6, label %bb433.i
	]
bb418.i:		; preds = %bb.i179
	br label %bb559.i
bb433.i:		; preds = %bb.i179
	switch i32 0, label %bb493.i [
		 i32 31744, label %bb455.i
		 i32 0, label %bb471.i
	]
bb455.i:		; preds = %bb433.i
	br i1 false, label %bb463.i, label %bb504.i
bb463.i:		; preds = %bb455.i
	br label %bb559.i
bb471.i:		; preds = %bb433.i
	br i1 false, label %bb497.i, label %bb484.preheader.i
bb484.preheader.i:		; preds = %bb471.i
	br i1 false, label %bb479.i, label %bb490.i
bb479.i:		; preds = %bb479.i, %bb484.preheader.i
	br i1 false, label %bb479.i, label %bb490.i
bb490.i:		; preds = %bb479.i, %bb484.preheader.i
	br label %bb559.i
bb493.i:		; preds = %bb433.i
	br label %bb497.i
bb497.i:		; preds = %bb493.i, %bb471.i
	br label %bb504.i
bb504.i:		; preds = %bb497.i, %bb455.i
	br label %bb513.i
bb513.i:		; preds = %bb504.i, %bb.i179
	br label %bb559.i
bb559.i:		; preds = %bb513.i, %bb490.i, %bb463.i, %bb418.i
	br i1 false, label %bb2793.i, label %bb614.i
bb614.i:		; preds = %bb559.i
	br i1 false, label %bb626.i, label %bb620.i
bb620.i:		; preds = %bb614.i
	br i1 false, label %bb625.i, label %bb626.i
bb625.i:		; preds = %bb620.i
	br label %bb626.i
bb626.i:		; preds = %bb625.i, %bb620.i, %bb614.i
	br i1 false, label %bb638.i, label %bb632.i
bb632.i:		; preds = %bb626.i
	br i1 false, label %bb637.i, label %bb638.i
bb637.i:		; preds = %bb632.i
	br label %bb638.i
bb638.i:		; preds = %bb637.i, %bb632.i, %bb626.i
	br i1 false, label %bb650.i, label %bb644.i
bb644.i:		; preds = %bb638.i
	br i1 false, label %bb649.i, label %bb650.i
bb649.i:		; preds = %bb644.i
	br label %bb650.i
bb650.i:		; preds = %bb649.i, %bb644.i, %bb638.i
	br i1 false, label %bb2793.i, label %bb656.i
bb656.i:		; preds = %bb650.i
	br i1 false, label %bb661.i, label %bb2793.i
bb661.i:		; preds = %bb656.i
	switch i32 0, label %bb2883.i [
		 i32 3, label %bb2874.i
		 i32 4, label %bb2795.i
		 i32 8, label %bb2810.i
		 i32 10, label %bb2834.i
		 i32 11, label %bb2819.i
		 i32 16, label %bb2810.i
	]
bb662.i:		; preds = %bb16671.i
	switch i32 0, label %bb1937.i [
		 i32 3, label %bb902.i
		 i32 4, label %bb1416.i
		 i32 8, label %bb1020.i
		 i32 10, label %bb902.i
		 i32 11, label %bb784.i
		 i32 16, label %bb664.i
	]
bb664.i:		; preds = %bb662.i
	br i1 false, label %bb682.i, label %bb669.i
bb669.i:		; preds = %bb664.i
	br label %bb710.i
bb682.i:		; preds = %bb664.i
	br label %bb710.i
bb710.i:		; preds = %bb682.i, %bb669.i
	br i1 false, label %bb760.i, label %bb754.i
bb754.i:		; preds = %bb710.i
	br i1 false, label %bb759.i, label %bb760.i
bb759.i:		; preds = %bb754.i
	br label %bb760.i
bb760.i:		; preds = %bb759.i, %bb754.i, %bb710.i
	br i1 false, label %bb772.i, label %bb766.i
bb766.i:		; preds = %bb760.i
	br i1 false, label %bb771.i, label %bb772.i
bb771.i:		; preds = %bb766.i
	br label %bb772.i
bb772.i:		; preds = %bb771.i, %bb766.i, %bb760.i
	br i1 false, label %bb1937.i, label %bb778.i
bb778.i:		; preds = %bb772.i
	br i1 false, label %bb783.i, label %bb1937.i
bb783.i:		; preds = %bb778.i
	br label %bb1937.i
bb784.i:		; preds = %bb662.i
	switch i32 0, label %bb892.i [
		 i32 1, label %bb868.i
		 i32 3, label %bb868.i
		 i32 4, label %bb882.i
		 i32 6, label %bb792.i
		 i32 7, label %bb786.i
	]
bb786.i:		; preds = %bb784.i
	br label %bb904.i
bb792.i:		; preds = %bb784.i
	switch i32 0, label %bb852.i [
		 i32 31744, label %bb814.i
		 i32 0, label %bb830.i
	]
bb814.i:		; preds = %bb792.i
	br i1 false, label %bb822.i, label %bb863.i
bb822.i:		; preds = %bb814.i
	switch i32 0, label %bb1010.i [
		 i32 1, label %bb986.i
		 i32 3, label %bb986.i
		 i32 4, label %bb1000.i
		 i32 6, label %bb910.i
		 i32 7, label %bb904.i
	]
bb830.i:		; preds = %bb792.i
	br i1 false, label %bb856.i, label %bb843.preheader.i
bb843.preheader.i:		; preds = %bb830.i
	br i1 false, label %bb838.i, label %bb849.i
bb838.i:		; preds = %bb838.i, %bb843.preheader.i
	br i1 false, label %bb838.i, label %bb849.i
bb849.i:		; preds = %bb838.i, %bb843.preheader.i
	switch i32 0, label %bb1010.i [
		 i32 1, label %bb986.i
		 i32 3, label %bb986.i
		 i32 4, label %bb1000.i
		 i32 6, label %bb910.i
		 i32 7, label %bb904.i
	]
bb852.i:		; preds = %bb792.i
	br label %bb856.i
bb856.i:		; preds = %bb852.i, %bb830.i
	switch i32 0, label %bb1010.i [
		 i32 1, label %bb986.i
		 i32 3, label %bb986.i
		 i32 4, label %bb1000.i
		 i32 6, label %bb910.i
		 i32 7, label %bb904.i
	]
bb863.i:		; preds = %bb814.i
	switch i32 0, label %bb1010.i [
		 i32 1, label %bb986.i
		 i32 3, label %bb986.i
		 i32 4, label %bb1000.i
		 i32 6, label %bb910.i
		 i32 7, label %bb904.i
	]
bb868.i:		; preds = %bb784.i, %bb784.i
	switch i32 0, label %bb1010.i [
		 i32 1, label %bb986.i
		 i32 3, label %bb986.i
		 i32 4, label %bb1000.i
		 i32 6, label %bb910.i
		 i32 7, label %bb904.i
	]
bb882.i:		; preds = %bb784.i
	br label %bb1000.i
bb892.i:		; preds = %bb784.i
	br label %bb902.i
bb902.i:		; preds = %bb892.i, %bb662.i, %bb662.i
	switch i32 0, label %bb1010.i [
		 i32 1, label %bb986.i
		 i32 3, label %bb986.i
		 i32 4, label %bb1000.i
		 i32 6, label %bb910.i
		 i32 7, label %bb904.i
	]
bb904.i:		; preds = %bb902.i, %bb868.i, %bb863.i, %bb856.i, %bb849.i, %bb822.i, %bb786.i
	br label %bb1937.i
bb910.i:		; preds = %bb902.i, %bb868.i, %bb863.i, %bb856.i, %bb849.i, %bb822.i
	switch i32 0, label %bb970.i [
		 i32 31744, label %bb932.i
		 i32 0, label %bb948.i
	]
bb932.i:		; preds = %bb910.i
	br i1 false, label %bb940.i, label %bb981.i
bb940.i:		; preds = %bb932.i
	br label %bb1937.i
bb948.i:		; preds = %bb910.i
	br i1 false, label %bb974.i, label %bb961.preheader.i
bb961.preheader.i:		; preds = %bb948.i
	br i1 false, label %bb956.i, label %bb967.i
bb956.i:		; preds = %bb956.i, %bb961.preheader.i
	br i1 false, label %bb956.i, label %bb967.i
bb967.i:		; preds = %bb956.i, %bb961.preheader.i
	br label %bb1937.i
bb970.i:		; preds = %bb910.i
	br label %bb974.i
bb974.i:		; preds = %bb970.i, %bb948.i
	br label %bb1937.i
bb981.i:		; preds = %bb932.i
	br label %bb1937.i
bb986.i:		; preds = %bb902.i, %bb902.i, %bb868.i, %bb868.i, %bb863.i, %bb863.i, %bb856.i, %bb856.i, %bb849.i, %bb849.i, %bb822.i, %bb822.i
	br label %bb1937.i
bb1000.i:		; preds = %bb902.i, %bb882.i, %bb868.i, %bb863.i, %bb856.i, %bb849.i, %bb822.i
	br label %bb1937.i
bb1010.i:		; preds = %bb902.i, %bb868.i, %bb863.i, %bb856.i, %bb849.i, %bb822.i
	br label %bb1937.i
bb1020.i:		; preds = %bb662.i
	switch i32 0, label %bb1388.i [
		 i32 1, label %bb1264.i
		 i32 3, label %bb1264.i
		 i32 4, label %bb1304.i
		 i32 6, label %bb1038.i
		 i32 7, label %bb1022.i
		 i32 8, label %bb1332.i
		 i32 9, label %bb1332.i
		 i32 10, label %bb1360.i
		 i32 11, label %bb1360.i
	]
bb1022.i:		; preds = %bb1020.i
	br label %bb1937.i
bb1038.i:		; preds = %bb1020.i
	switch i32 0, label %bb1098.i [
		 i32 31744, label %bb1060.i
		 i32 0, label %bb1076.i
	]
bb1060.i:		; preds = %bb1038.i
	br i1 false, label %bb1068.i, label %bb1109.i
bb1068.i:		; preds = %bb1060.i
	br label %bb1109.i
bb1076.i:		; preds = %bb1038.i
	br i1 false, label %bb1102.i, label %bb1089.preheader.i
bb1089.preheader.i:		; preds = %bb1076.i
	br i1 false, label %bb1084.i, label %bb1095.i
bb1084.i:		; preds = %bb1084.i, %bb1089.preheader.i
	br i1 false, label %bb1084.i, label %bb1095.i
bb1095.i:		; preds = %bb1084.i, %bb1089.preheader.i
	br label %bb1109.i
bb1098.i:		; preds = %bb1038.i
	br label %bb1102.i
bb1102.i:		; preds = %bb1098.i, %bb1076.i
	br label %bb1109.i
bb1109.i:		; preds = %bb1102.i, %bb1095.i, %bb1068.i, %bb1060.i
	switch i32 0, label %bb1173.i [
		 i32 31744, label %bb1135.i
		 i32 0, label %bb1151.i
	]
bb1135.i:		; preds = %bb1109.i
	br i1 false, label %bb1143.i, label %bb1184.i
bb1143.i:		; preds = %bb1135.i
	br label %bb1184.i
bb1151.i:		; preds = %bb1109.i
	br i1 false, label %bb1177.i, label %bb1164.preheader.i
bb1164.preheader.i:		; preds = %bb1151.i
	br i1 false, label %bb1159.i, label %bb1170.i
bb1159.i:		; preds = %bb1159.i, %bb1164.preheader.i
	br i1 false, label %bb1159.i, label %bb1170.i
bb1170.i:		; preds = %bb1159.i, %bb1164.preheader.i
	br label %bb1184.i
bb1173.i:		; preds = %bb1109.i
	br label %bb1177.i
bb1177.i:		; preds = %bb1173.i, %bb1151.i
	br label %bb1184.i
bb1184.i:		; preds = %bb1177.i, %bb1170.i, %bb1143.i, %bb1135.i
	switch i32 0, label %bb1248.i [
		 i32 31744, label %bb1210.i
		 i32 0, label %bb1226.i
	]
bb1210.i:		; preds = %bb1184.i
	br i1 false, label %bb1218.i, label %bb1259.i
bb1218.i:		; preds = %bb1210.i
	br label %bb1937.i
bb1226.i:		; preds = %bb1184.i
	br i1 false, label %bb1252.i, label %bb1239.preheader.i
bb1239.preheader.i:		; preds = %bb1226.i
	br i1 false, label %bb1234.i, label %bb1245.i
bb1234.i:		; preds = %bb1234.i, %bb1239.preheader.i
	br i1 false, label %bb1234.i, label %bb1245.i
bb1245.i:		; preds = %bb1234.i, %bb1239.preheader.i
	br label %bb1937.i
bb1248.i:		; preds = %bb1184.i
	br label %bb1252.i
bb1252.i:		; preds = %bb1248.i, %bb1226.i
	br label %bb1937.i
bb1259.i:		; preds = %bb1210.i
	br label %bb1937.i
bb1264.i:		; preds = %bb1020.i, %bb1020.i
	br label %bb1937.i
bb1304.i:		; preds = %bb1020.i
	br label %bb1937.i
bb1332.i:		; preds = %bb1020.i, %bb1020.i
	br label %bb1937.i
bb1360.i:		; preds = %bb1020.i, %bb1020.i
	br label %bb1937.i
bb1388.i:		; preds = %bb1020.i
	br label %bb1937.i
bb1416.i:		; preds = %bb662.i
	switch i32 0, label %bb1900.i [
		 i32 1, label %bb1740.i
		 i32 3, label %bb1740.i
		 i32 4, label %bb1793.i
		 i32 6, label %bb1439.i
		 i32 7, label %bb1418.i
		 i32 14, label %bb1830.i
		 i32 15, label %bb1830.i
		 i32 18, label %bb1863.i
		 i32 19, label %bb1863.i
	]
bb1418.i:		; preds = %bb1416.i
	br label %bb1937.i
bb1439.i:		; preds = %bb1416.i
	switch i32 0, label %bb1499.i [
		 i32 31744, label %bb1461.i
		 i32 0, label %bb1477.i
	]
bb1461.i:		; preds = %bb1439.i
	br i1 false, label %bb1469.i, label %bb1510.i
bb1469.i:		; preds = %bb1461.i
	br label %bb1510.i
bb1477.i:		; preds = %bb1439.i
	br i1 false, label %bb1503.i, label %bb1490.preheader.i
bb1490.preheader.i:		; preds = %bb1477.i
	br i1 false, label %bb1485.i, label %bb1496.i
bb1485.i:		; preds = %bb1485.i, %bb1490.preheader.i
	br i1 false, label %bb1485.i, label %bb1496.i
bb1496.i:		; preds = %bb1485.i, %bb1490.preheader.i
	br label %bb1510.i
bb1499.i:		; preds = %bb1439.i
	br label %bb1503.i
bb1503.i:		; preds = %bb1499.i, %bb1477.i
	br label %bb1510.i
bb1510.i:		; preds = %bb1503.i, %bb1496.i, %bb1469.i, %bb1461.i
	switch i32 0, label %bb1574.i [
		 i32 31744, label %bb1536.i
		 i32 0, label %bb1552.i
	]
bb1536.i:		; preds = %bb1510.i
	br i1 false, label %bb1544.i, label %bb1585.i
bb1544.i:		; preds = %bb1536.i
	br label %bb1585.i
bb1552.i:		; preds = %bb1510.i
	br i1 false, label %bb1578.i, label %bb1565.preheader.i
bb1565.preheader.i:		; preds = %bb1552.i
	br i1 false, label %bb1560.i, label %bb1571.i
bb1560.i:		; preds = %bb1560.i, %bb1565.preheader.i
	br i1 false, label %bb1560.i, label %bb1571.i
bb1571.i:		; preds = %bb1560.i, %bb1565.preheader.i
	br label %bb1585.i
bb1574.i:		; preds = %bb1510.i
	br label %bb1578.i
bb1578.i:		; preds = %bb1574.i, %bb1552.i
	br label %bb1585.i
bb1585.i:		; preds = %bb1578.i, %bb1571.i, %bb1544.i, %bb1536.i
	switch i32 0, label %bb1649.i [
		 i32 31744, label %bb1611.i
		 i32 0, label %bb1627.i
	]
bb1611.i:		; preds = %bb1585.i
	br i1 false, label %bb1619.i, label %bb1660.i
bb1619.i:		; preds = %bb1611.i
	br label %bb1660.i
bb1627.i:		; preds = %bb1585.i
	br i1 false, label %bb1653.i, label %bb1640.preheader.i
bb1640.preheader.i:		; preds = %bb1627.i
	br i1 false, label %bb1635.i, label %bb1646.i
bb1635.i:		; preds = %bb1635.i, %bb1640.preheader.i
	br i1 false, label %bb1635.i, label %bb1646.i
bb1646.i:		; preds = %bb1635.i, %bb1640.preheader.i
	br label %bb1660.i
bb1649.i:		; preds = %bb1585.i
	br label %bb1653.i
bb1653.i:		; preds = %bb1649.i, %bb1627.i
	br label %bb1660.i
bb1660.i:		; preds = %bb1653.i, %bb1646.i, %bb1619.i, %bb1611.i
	switch i32 0, label %bb1724.i [
		 i32 31744, label %bb1686.i
		 i32 0, label %bb1702.i
	]
bb1686.i:		; preds = %bb1660.i
	br i1 false, label %bb1694.i, label %bb1735.i
bb1694.i:		; preds = %bb1686.i
	br label %bb1937.i
bb1702.i:		; preds = %bb1660.i
	br i1 false, label %bb1728.i, label %bb1715.preheader.i
bb1715.preheader.i:		; preds = %bb1702.i
	br i1 false, label %bb1710.i, label %bb1721.i
bb1710.i:		; preds = %bb1710.i, %bb1715.preheader.i
	br i1 false, label %bb1710.i, label %bb1721.i
bb1721.i:		; preds = %bb1710.i, %bb1715.preheader.i
	br label %bb1937.i
bb1724.i:		; preds = %bb1660.i
	br label %bb1728.i
bb1728.i:		; preds = %bb1724.i, %bb1702.i
	br label %bb1937.i
bb1735.i:		; preds = %bb1686.i
	br label %bb1937.i
bb1740.i:		; preds = %bb1416.i, %bb1416.i
	br label %bb1937.i
bb1793.i:		; preds = %bb1416.i
	br label %bb1937.i
bb1830.i:		; preds = %bb1416.i, %bb1416.i
	br label %bb1937.i
bb1863.i:		; preds = %bb1416.i, %bb1416.i
	br label %bb1937.i
bb1900.i:		; preds = %bb1416.i
	br label %bb1937.i
bb1937.i:		; preds = %bb1900.i, %bb1863.i, %bb1830.i, %bb1793.i, %bb1740.i, %bb1735.i, %bb1728.i, %bb1721.i, %bb1694.i, %bb1418.i, %bb1388.i, %bb1360.i, %bb1332.i, %bb1304.i, %bb1264.i, %bb1259.i, %bb1252.i, %bb1245.i, %bb1218.i, %bb1022.i, %bb1010.i, %bb1000.i, %bb986.i, %bb981.i, %bb974.i, %bb967.i, %bb940.i, %bb904.i, %bb783.i, %bb778.i, %bb772.i, %bb662.i
	switch i32 %sf4083.0.i, label %bb2321.i [
		 i32 0, label %bb2027.i
		 i32 1, label %bb2081.i
		 i32 2, label %bb2161.i
		 i32 3, label %bb2241.i
		 i32 8, label %bb1939.i
		 i32 9, label %bb1939.i
		 i32 10, label %bb1957.i
		 i32 11, label %bb1975.i
		 i32 16, label %bb1939.i
	]
bb1939.i:		; preds = %bb1937.i, %bb1937.i, %bb1937.i
	switch i32 0, label %bb2321.i [
		 i32 3, label %bb1956.i
		 i32 4, label %bb1956.i
		 i32 11, label %bb1956.i
	]
bb1956.i:		; preds = %bb1939.i, %bb1939.i, %bb1939.i
	br label %bb2337.i
bb1957.i:		; preds = %bb1937.i
	switch i32 0, label %bb1975.i [
		 i32 3, label %bb1974.i
		 i32 4, label %bb1974.i
		 i32 11, label %bb1974.i
	]
bb1974.i:		; preds = %bb1957.i, %bb1957.i, %bb1957.i
	br label %bb1975.i
bb1975.i:		; preds = %bb1974.i, %bb1957.i, %bb1937.i
	switch i32 0, label %bb2001.i [
		 i32 1, label %bb1992.i
		 i32 4, label %bb1992.i
		 i32 8, label %bb1992.i
	]
bb1992.i:		; preds = %bb1975.i, %bb1975.i, %bb1975.i
	br label %bb2001.i
bb2001.i:		; preds = %bb1992.i, %bb1975.i
	switch i32 0, label %bb2321.i [
		 i32 2, label %bb2018.i
		 i32 4, label %bb2018.i
		 i32 8, label %bb2018.i
	]
bb2018.i:		; preds = %bb2001.i, %bb2001.i, %bb2001.i
	br label %bb2321.i
bb2027.i:		; preds = %bb1937.i
	switch i32 0, label %bb2045.i [
		 i32 1, label %bb2044.i
		 i32 4, label %bb2044.i
		 i32 8, label %bb2044.i
	]
bb2044.i:		; preds = %bb2027.i, %bb2027.i, %bb2027.i
	br label %bb2045.i
bb2045.i:		; preds = %bb2044.i, %bb2027.i
	switch i32 0, label %bb2063.i [
		 i32 2, label %bb2062.i
		 i32 4, label %bb2062.i
		 i32 8, label %bb2062.i
	]
bb2062.i:		; preds = %bb2045.i, %bb2045.i, %bb2045.i
	br label %bb2063.i
bb2063.i:		; preds = %bb2062.i, %bb2045.i
	switch i32 0, label %bb2321.i [
		 i32 3, label %bb2080.i
		 i32 4, label %bb2080.i
		 i32 11, label %bb2080.i
	]
bb2080.i:		; preds = %bb2063.i, %bb2063.i, %bb2063.i
	br label %bb2321.i
bb2081.i:		; preds = %bb1937.i
	switch i32 0, label %bb2100.i [
		 i32 1, label %bb2098.i
		 i32 4, label %bb2098.i
		 i32 8, label %bb2098.i
	]
bb2098.i:		; preds = %bb2081.i, %bb2081.i, %bb2081.i
	br label %bb2100.i
bb2100.i:		; preds = %bb2098.i, %bb2081.i
	switch i32 0, label %bb2125.i [
		 i32 4, label %bb2124.i
		 i32 8, label %bb2124.i
		 i32 0, label %bb2124.i
		 i32 11, label %bb2124.i
	]
bb2124.i:		; preds = %bb2100.i, %bb2100.i, %bb2100.i, %bb2100.i
	br label %bb2125.i
bb2125.i:		; preds = %bb2124.i, %bb2100.i
	switch i32 0, label %bb2143.i [
		 i32 2, label %bb2142.i
		 i32 4, label %bb2142.i
		 i32 8, label %bb2142.i
	]
bb2142.i:		; preds = %bb2125.i, %bb2125.i, %bb2125.i
	br label %bb2143.i
bb2143.i:		; preds = %bb2142.i, %bb2125.i
	switch i32 0, label %bb2321.i [
		 i32 3, label %bb2160.i
		 i32 4, label %bb2160.i
		 i32 11, label %bb2160.i
	]
bb2160.i:		; preds = %bb2143.i, %bb2143.i, %bb2143.i
	br label %bb2321.i
bb2161.i:		; preds = %bb1937.i
	switch i32 0, label %bb2180.i [
		 i32 2, label %bb2178.i
		 i32 4, label %bb2178.i
		 i32 8, label %bb2178.i
	]
bb2178.i:		; preds = %bb2161.i, %bb2161.i, %bb2161.i
	br label %bb2180.i
bb2180.i:		; preds = %bb2178.i, %bb2161.i
	switch i32 0, label %bb2205.i [
		 i32 4, label %bb2204.i
		 i32 8, label %bb2204.i
		 i32 0, label %bb2204.i
		 i32 11, label %bb2204.i
	]
bb2204.i:		; preds = %bb2180.i, %bb2180.i, %bb2180.i, %bb2180.i
	br label %bb2205.i
bb2205.i:		; preds = %bb2204.i, %bb2180.i
	switch i32 0, label %bb2223.i [
		 i32 1, label %bb2222.i
		 i32 4, label %bb2222.i
		 i32 8, label %bb2222.i
	]
bb2222.i:		; preds = %bb2205.i, %bb2205.i, %bb2205.i
	br label %bb2223.i
bb2223.i:		; preds = %bb2222.i, %bb2205.i
	switch i32 0, label %bb2321.i [
		 i32 3, label %bb2240.i
		 i32 4, label %bb2240.i
		 i32 11, label %bb2240.i
	]
bb2240.i:		; preds = %bb2223.i, %bb2223.i, %bb2223.i
	br label %bb2321.i
bb2241.i:		; preds = %bb1937.i
	switch i32 0, label %bb2260.i [
		 i32 3, label %bb2258.i
		 i32 4, label %bb2258.i
		 i32 11, label %bb2258.i
	]
bb2258.i:		; preds = %bb2241.i, %bb2241.i, %bb2241.i
	br label %bb2260.i
bb2260.i:		; preds = %bb2258.i, %bb2241.i
	switch i32 0, label %bb2285.i [
		 i32 4, label %bb2284.i
		 i32 11, label %bb2284.i
		 i32 0, label %bb2284.i
		 i32 8, label %bb2284.i
	]
bb2284.i:		; preds = %bb2260.i, %bb2260.i, %bb2260.i, %bb2260.i
	br label %bb2285.i
bb2285.i:		; preds = %bb2284.i, %bb2260.i
	switch i32 0, label %bb2303.i [
		 i32 1, label %bb2302.i
		 i32 4, label %bb2302.i
		 i32 8, label %bb2302.i
	]
bb2302.i:		; preds = %bb2285.i, %bb2285.i, %bb2285.i
	br label %bb2303.i
bb2303.i:		; preds = %bb2302.i, %bb2285.i
	switch i32 0, label %bb2321.i [
		 i32 2, label %bb2320.i
		 i32 4, label %bb2320.i
		 i32 8, label %bb2320.i
	]
bb2320.i:		; preds = %bb2303.i, %bb2303.i, %bb2303.i
	br label %bb2321.i
bb2321.i:		; preds = %bb2320.i, %bb2303.i, %bb2240.i, %bb2223.i, %bb2160.i, %bb2143.i, %bb2080.i, %bb2063.i, %bb2018.i, %bb2001.i, %bb1939.i, %bb1937.i
	br label %bb2337.i
bb2337.i:		; preds = %bb2321.i, %bb1956.i
	br label %bb2353.i
bb2353.i:		; preds = %bb2337.i
	br label %bb2369.i
bb2369.i:		; preds = %bb2353.i
	br label %bb2385.i
bb2385.i:		; preds = %bb2369.i
	br i1 false, label %bb2388.i, label %bb2394.i
bb2388.i:		; preds = %bb2385.i
	br label %bb2600.i
bb2394.i:		; preds = %bb2385.i
	switch i32 0, label %bb2600.i [
		 i32 0, label %bb2504.i
		 i32 1, label %bb2528.i
		 i32 2, label %bb2552.i
		 i32 3, label %bb2576.i
		 i32 4, label %bb2396.i
		 i32 8, label %bb2420.i
		 i32 11, label %bb2480.i
	]
bb2396.i:		; preds = %bb2394.i
	br i1 false, label %bb2411.i, label %bb2399.i
bb2399.i:		; preds = %bb2396.i
	br i1 false, label %bb2420.i, label %bb2405.i
bb2405.i:		; preds = %bb2399.i
	br i1 false, label %bb2410.i, label %bb2420.i
bb2410.i:		; preds = %bb2405.i
	br i1 false, label %bb2459.i, label %bb2423.i
bb2411.i:		; preds = %bb2396.i
	br i1 false, label %bb2420.i, label %bb2414.i
bb2414.i:		; preds = %bb2411.i
	br i1 false, label %bb2419.i, label %bb2420.i
bb2419.i:		; preds = %bb2414.i
	br label %bb2420.i
bb2420.i:		; preds = %bb2419.i, %bb2414.i, %bb2411.i, %bb2405.i, %bb2399.i, %bb2394.i
	br i1 false, label %bb2459.i, label %bb2423.i
bb2423.i:		; preds = %bb2420.i, %bb2410.i
	br i1 false, label %bb2435.i, label %bb2429.i
bb2429.i:		; preds = %bb2423.i
	br i1 false, label %bb2434.i, label %bb2435.i
bb2434.i:		; preds = %bb2429.i
	br label %bb2435.i
bb2435.i:		; preds = %bb2434.i, %bb2429.i, %bb2423.i
	br i1 false, label %bb2447.i, label %bb2441.i
bb2441.i:		; preds = %bb2435.i
	br i1 false, label %bb2446.i, label %bb2447.i
bb2446.i:		; preds = %bb2441.i
	br label %bb2447.i
bb2447.i:		; preds = %bb2446.i, %bb2441.i, %bb2435.i
	br i1 false, label %bb2600.i, label %bb2453.i
bb2453.i:		; preds = %bb2447.i
	br i1 false, label %bb2458.i, label %bb2600.i
bb2458.i:		; preds = %bb2453.i
	br label %bb2793.i
bb2459.i:		; preds = %bb2420.i, %bb2410.i
	br i1 false, label %bb2600.i, label %bb2462.i
bb2462.i:		; preds = %bb2459.i
	br i1 false, label %bb2479.i, label %bb2600.i
bb2479.i:		; preds = %bb2462.i
	br label %bb2600.i
bb2480.i:		; preds = %bb2394.i
	br i1 false, label %bb2495.i, label %bb2483.i
bb2483.i:		; preds = %bb2480.i
	br i1 false, label %bb2504.i, label %bb2489.i
bb2489.i:		; preds = %bb2483.i
	br i1 false, label %bb2494.i, label %bb2504.i
bb2494.i:		; preds = %bb2489.i
	br i1 false, label %bb2519.i, label %bb2507.i
bb2495.i:		; preds = %bb2480.i
	br i1 false, label %bb2504.i, label %bb2498.i
bb2498.i:		; preds = %bb2495.i
	br i1 false, label %bb2503.i, label %bb2504.i
bb2503.i:		; preds = %bb2498.i
	br label %bb2504.i
bb2504.i:		; preds = %bb2503.i, %bb2498.i, %bb2495.i, %bb2489.i, %bb2483.i, %bb2394.i
	br i1 false, label %bb2519.i, label %bb2507.i
bb2507.i:		; preds = %bb2504.i, %bb2494.i
	br i1 false, label %bb2600.i, label %bb2513.i
bb2513.i:		; preds = %bb2507.i
	br i1 false, label %bb2518.i, label %bb2600.i
bb2518.i:		; preds = %bb2513.i
	br label %bb2600.i
bb2519.i:		; preds = %bb2504.i, %bb2494.i
	br i1 false, label %bb2600.i, label %bb2522.i
bb2522.i:		; preds = %bb2519.i
	br i1 false, label %bb2527.i, label %bb2600.i
bb2527.i:		; preds = %bb2522.i
	br label %bb2600.i
bb2528.i:		; preds = %bb2394.i
	br i1 false, label %bb2543.i, label %bb2531.i
bb2531.i:		; preds = %bb2528.i
	br i1 false, label %bb2600.i, label %bb2537.i
bb2537.i:		; preds = %bb2531.i
	br i1 false, label %bb2542.i, label %bb2600.i
bb2542.i:		; preds = %bb2537.i
	br label %bb2600.i
bb2543.i:		; preds = %bb2528.i
	br i1 false, label %bb2600.i, label %bb2546.i
bb2546.i:		; preds = %bb2543.i
	br i1 false, label %bb2551.i, label %bb2600.i
bb2551.i:		; preds = %bb2546.i
	br label %bb2600.i
bb2552.i:		; preds = %bb2394.i
	br i1 false, label %bb2567.i, label %bb2555.i
bb2555.i:		; preds = %bb2552.i
	br i1 false, label %bb2600.i, label %bb2561.i
bb2561.i:		; preds = %bb2555.i
	br i1 false, label %bb2566.i, label %bb2600.i
bb2566.i:		; preds = %bb2561.i
	br label %bb2600.i
bb2567.i:		; preds = %bb2552.i
	br i1 false, label %bb2600.i, label %bb2570.i
bb2570.i:		; preds = %bb2567.i
	br i1 false, label %bb2575.i, label %bb2600.i
bb2575.i:		; preds = %bb2570.i
	br label %bb2600.i
bb2576.i:		; preds = %bb2394.i
	br i1 false, label %bb2591.i, label %bb2579.i
bb2579.i:		; preds = %bb2576.i
	br i1 false, label %bb2600.i, label %bb2585.i
bb2585.i:		; preds = %bb2579.i
	br i1 false, label %bb2590.i, label %bb2600.i
bb2590.i:		; preds = %bb2585.i
	br label %bb2600.i
bb2591.i:		; preds = %bb2576.i
	br i1 false, label %bb2600.i, label %bb2594.i
bb2594.i:		; preds = %bb2591.i
	br i1 false, label %bb2599.i, label %bb2600.i
bb2599.i:		; preds = %bb2594.i
	br label %bb2600.i
bb2600.i:		; preds = %bb2599.i, %bb2594.i, %bb2591.i, %bb2590.i, %bb2585.i, %bb2579.i, %bb2575.i, %bb2570.i, %bb2567.i, %bb2566.i, %bb2561.i, %bb2555.i, %bb2551.i, %bb2546.i, %bb2543.i, %bb2542.i, %bb2537.i, %bb2531.i, %bb2527.i, %bb2522.i, %bb2519.i, %bb2518.i, %bb2513.i, %bb2507.i, %bb2479.i, %bb2462.i, %bb2459.i, %bb2453.i, %bb2447.i, %bb2394.i, %bb2388.i
	br label %bb2793.i
bb2793.i:		; preds = %bb2600.i, %bb2458.i, %bb656.i, %bb650.i, %bb559.i
	switch i32 0, label %bb2883.i [
		 i32 3, label %bb2874.i
		 i32 4, label %bb2795.i
		 i32 8, label %bb2810.i
		 i32 10, label %bb2834.i
		 i32 11, label %bb2819.i
		 i32 16, label %bb2810.i
	]
bb2795.i:		; preds = %bb2793.i, %bb661.i
	br label %bb2810.i
bb2810.i:		; preds = %bb2795.i, %bb2793.i, %bb2793.i, %bb661.i, %bb661.i
	br label %bb2883.i
bb2819.i:		; preds = %bb2793.i, %bb661.i
	br label %bb2834.i
bb2834.i:		; preds = %bb2819.i, %bb2793.i, %bb661.i
	switch i32 0, label %bb2860.i [
		 i32 4, label %bb2846.i
		 i32 8, label %bb2846.i
	]
bb2846.i:		; preds = %bb2834.i, %bb2834.i
	br i1 false, label %bb2859.i, label %bb2860.i
bb2859.i:		; preds = %bb2846.i
	br label %bb2860.i
bb2860.i:		; preds = %bb2859.i, %bb2846.i, %bb2834.i
	switch i32 %df4081.0.i, label %bb2867.bb2883_crit_edge.i [
		 i32 1, label %bb2883.i
		 i32 2, label %bb2872.i
	]
bb2867.bb2883_crit_edge.i:		; preds = %bb2860.i
	br label %bb2883.i
bb2872.i:		; preds = %bb2860.i
	switch i32 0, label %UnifiedReturnBlock.i235 [
		 i32 3, label %bb3253.i
		 i32 4, label %bb4173.i
		 i32 8, label %bb3485.i
		 i32 10, label %bb3253.i
		 i32 11, label %bb3021.i
		 i32 16, label %bb2885.i
	]
bb2874.i:		; preds = %bb2793.i, %bb661.i
	br label %bb2883.i
bb2883.i:		; preds = %bb2874.i, %bb2867.bb2883_crit_edge.i, %bb2860.i, %bb2810.i, %bb2793.i, %bb661.i
	%f_alpha.1.i = phi i32 [ 0, %bb2867.bb2883_crit_edge.i ], [ 0, %bb2874.i ], [ 1065353216, %bb661.i ], [ 0, %bb2793.i ], [ 0, %bb2810.i ], [ 0, %bb2860.i ]		; <i32> [#uses=1]
	switch i32 0, label %UnifiedReturnBlock.i235 [
		 i32 3, label %bb3253.i
		 i32 4, label %bb4173.i
		 i32 8, label %bb3485.i
		 i32 10, label %bb3253.i
		 i32 11, label %bb3021.i
		 i32 16, label %bb2885.i
	]
bb2885.i:		; preds = %bb2883.i, %bb2872.i
	br i1 false, label %bb3011.i, label %bb2890.i
bb2890.i:		; preds = %bb2885.i
	br i1 false, label %bb2960.i, label %bb2954.i
bb2954.i:		; preds = %bb2890.i
	br i1 false, label %bb2959.i, label %bb2960.i
bb2959.i:		; preds = %bb2954.i
	br label %bb2960.i
bb2960.i:		; preds = %bb2959.i, %bb2954.i, %bb2890.i
	br i1 false, label %bb2972.i, label %bb2966.i
bb2966.i:		; preds = %bb2960.i
	br i1 false, label %bb2971.i, label %bb2972.i
bb2971.i:		; preds = %bb2966.i
	br label %bb2972.i
bb2972.i:		; preds = %bb2971.i, %bb2966.i, %bb2960.i
	br label %glgScalarFloatConversion.exit
bb3011.i:		; preds = %bb2885.i
	br label %glgScalarFloatConversion.exit
bb3021.i:		; preds = %bb2883.i, %bb2872.i
	switch i32 %dt4080.0.i, label %bb3192.i [
		 i32 7, label %bb3026.i
		 i32 6, label %bb3037.i
		 i32 1, label %bb3125.i
		 i32 3, label %bb3125.i
		 i32 5, label %bb3144.i
	]
bb3026.i:		; preds = %bb3021.i
	br label %bb3258.i
bb3037.i:		; preds = %bb3021.i
	br i1 false, label %bb3052.i, label %bb3074.i
bb3052.i:		; preds = %bb3037.i
	br i1 false, label %bb3105.i, label %bb3069.i
bb3069.i:		; preds = %bb3052.i
	switch i32 %dt4080.0.i, label %bb3424.i [
		 i32 7, label %bb3258.i
		 i32 6, label %bb3269.i
		 i32 1, label %bb3357.i
		 i32 3, label %bb3357.i
		 i32 5, label %bb3376.i
	]
bb3074.i:		; preds = %bb3037.i
	br i1 false, label %bb3079.i, label %bb3092.i
bb3079.i:		; preds = %bb3074.i
	switch i32 %dt4080.0.i, label %bb3424.i [
		 i32 7, label %bb3258.i
		 i32 6, label %bb3269.i
		 i32 1, label %bb3357.i
		 i32 3, label %bb3357.i
		 i32 5, label %bb3376.i
	]
bb3092.i:		; preds = %bb3074.i
	switch i32 %dt4080.0.i, label %bb3424.i [
		 i32 7, label %bb3258.i
		 i32 6, label %bb3269.i
		 i32 1, label %bb3357.i
		 i32 3, label %bb3357.i
		 i32 5, label %bb3376.i
	]
bb3105.i:		; preds = %bb3052.i
	switch i32 %dt4080.0.i, label %bb3424.i [
		 i32 7, label %bb3258.i
		 i32 6, label %bb3269.i
		 i32 1, label %bb3357.i
		 i32 3, label %bb3357.i
		 i32 5, label %bb3376.i
	]
bb3125.i:		; preds = %bb3021.i, %bb3021.i
	switch i32 %dt4080.0.i, label %bb3424.i [
		 i32 7, label %bb3258.i
		 i32 6, label %bb3269.i
		 i32 1, label %bb3357.i
		 i32 3, label %bb3357.i
		 i32 5, label %bb3376.i
	]
bb3144.i:		; preds = %bb3021.i
	br label %bb3376.i
bb3192.i:		; preds = %bb3021.i
	br i1 false, label %bb3197.i, label %bb3243.i
bb3197.i:		; preds = %bb3192.i
	br label %bb3424.i
bb3243.i:		; preds = %bb3192.i
	br label %bb3253.i
bb3253.i:		; preds = %bb3243.i, %bb2883.i, %bb2883.i, %bb2872.i, %bb2872.i
	switch i32 %dt4080.0.i, label %bb3424.i [
		 i32 7, label %bb3258.i
		 i32 6, label %bb3269.i
		 i32 1, label %bb3357.i
		 i32 3, label %bb3357.i
		 i32 5, label %bb3376.i
	]
bb3258.i:		; preds = %bb3253.i, %bb3125.i, %bb3105.i, %bb3092.i, %bb3079.i, %bb3069.i, %bb3026.i
	br label %glgScalarFloatConversion.exit
bb3269.i:		; preds = %bb3253.i, %bb3125.i, %bb3105.i, %bb3092.i, %bb3079.i, %bb3069.i
	br i1 false, label %bb3284.i, label %bb3306.i
bb3284.i:		; preds = %bb3269.i
	br i1 false, label %bb3337.i, label %bb3301.i
bb3301.i:		; preds = %bb3284.i
	br label %glgScalarFloatConversion.exit
bb3306.i:		; preds = %bb3269.i
	br i1 false, label %bb3311.i, label %bb3324.i
bb3311.i:		; preds = %bb3306.i
	br label %glgScalarFloatConversion.exit
bb3324.i:		; preds = %bb3306.i
	br label %glgScalarFloatConversion.exit
bb3337.i:		; preds = %bb3284.i
	br label %glgScalarFloatConversion.exit
bb3357.i:		; preds = %bb3253.i, %bb3253.i, %bb3125.i, %bb3125.i, %bb3105.i, %bb3105.i, %bb3092.i, %bb3092.i, %bb3079.i, %bb3079.i, %bb3069.i, %bb3069.i
	br label %glgScalarFloatConversion.exit
bb3376.i:		; preds = %bb3253.i, %bb3144.i, %bb3125.i, %bb3105.i, %bb3092.i, %bb3079.i, %bb3069.i
	br label %glgScalarFloatConversion.exit
bb3424.i:		; preds = %bb3253.i, %bb3197.i, %bb3125.i, %bb3105.i, %bb3092.i, %bb3079.i, %bb3069.i
	br i1 false, label %bb3429.i, label %bb3475.i
bb3429.i:		; preds = %bb3424.i
	br label %glgScalarFloatConversion.exit
bb3475.i:		; preds = %bb3424.i
	br label %glgScalarFloatConversion.exit
bb3485.i:		; preds = %bb2883.i, %bb2872.i
	switch i32 %dt4080.0.i, label %bb4077.i [
		 i32 7, label %bb3490.i
		 i32 6, label %bb3511.i
		 i32 1, label %bb3749.i
		 i32 3, label %bb3749.i
		 i32 5, label %bb3794.i
		 i32 4, label %bb3941.i
	]
bb3490.i:		; preds = %bb3485.i
	br label %glgScalarFloatConversion.exit
bb3511.i:		; preds = %bb3485.i
	br i1 false, label %bb3526.i, label %bb3548.i
bb3526.i:		; preds = %bb3511.i
	br i1 false, label %bb3579.i, label %bb3543.i
bb3543.i:		; preds = %bb3526.i
	br label %bb3579.i
bb3548.i:		; preds = %bb3511.i
	br i1 false, label %bb3553.i, label %bb3566.i
bb3553.i:		; preds = %bb3548.i
	br label %bb3579.i
bb3566.i:		; preds = %bb3548.i
	br label %bb3579.i
bb3579.i:		; preds = %bb3566.i, %bb3553.i, %bb3543.i, %bb3526.i
	br i1 false, label %bb3601.i, label %bb3623.i
bb3601.i:		; preds = %bb3579.i
	br i1 false, label %bb3654.i, label %bb3618.i
bb3618.i:		; preds = %bb3601.i
	br label %bb3654.i
bb3623.i:		; preds = %bb3579.i
	br i1 false, label %bb3628.i, label %bb3641.i
bb3628.i:		; preds = %bb3623.i
	br label %bb3654.i
bb3641.i:		; preds = %bb3623.i
	br label %bb3654.i
bb3654.i:		; preds = %bb3641.i, %bb3628.i, %bb3618.i, %bb3601.i
	br i1 false, label %bb3676.i, label %bb3698.i
bb3676.i:		; preds = %bb3654.i
	br i1 false, label %bb3729.i, label %bb3693.i
bb3693.i:		; preds = %bb3676.i
	br label %glgScalarFloatConversion.exit
bb3698.i:		; preds = %bb3654.i
	br i1 false, label %bb3703.i, label %bb3716.i
bb3703.i:		; preds = %bb3698.i
	br label %glgScalarFloatConversion.exit
bb3716.i:		; preds = %bb3698.i
	br label %glgScalarFloatConversion.exit
bb3729.i:		; preds = %bb3676.i
	br label %glgScalarFloatConversion.exit
bb3749.i:		; preds = %bb3485.i, %bb3485.i
	br label %glgScalarFloatConversion.exit
bb3794.i:		; preds = %bb3485.i
	br label %glgScalarFloatConversion.exit
bb3941.i:		; preds = %bb3485.i
	br label %glgScalarFloatConversion.exit
bb4077.i:		; preds = %bb3485.i
	br i1 false, label %bb4083.i, label %bb4111.i
bb4083.i:		; preds = %bb4077.i
	br label %glgScalarFloatConversion.exit
bb4111.i:		; preds = %bb4077.i
	br i1 false, label %bb4117.i, label %bb4145.i
bb4117.i:		; preds = %bb4111.i
	br label %glgScalarFloatConversion.exit
bb4145.i:		; preds = %bb4111.i
	br label %glgScalarFloatConversion.exit
bb4173.i:		; preds = %bb2883.i, %bb2872.i
	%f_red.0.reg2mem.4.i = phi i32 [ 0, %bb2872.i ], [ 0, %bb2883.i ]		; <i32> [#uses=2]
	%f_green.0.reg2mem.2.i = phi i32 [ 0, %bb2872.i ], [ 0, %bb2883.i ]		; <i32> [#uses=1]
	%f_blue.0.reg2mem.2.i = phi i32 [ 0, %bb2872.i ], [ 0, %bb2883.i ]		; <i32> [#uses=1]
	%f_alpha.1.reg2mem.1.i = phi i32 [ 0, %bb2872.i ], [ %f_alpha.1.i, %bb2883.i ]		; <i32> [#uses=1]
	switch i32 %dt4080.0.i, label %bb4950.i [
		 i32 7, label %bb4178.i
		 i32 6, label %bb4204.i
		 i32 1, label %bb4517.i202
		 i32 3, label %bb4517.i202
		 i32 5, label %bb4575.i
		 i32 4, label %bb4769.i
	]
bb4178.i:		; preds = %bb4173.i
	br label %glgScalarFloatConversion.exit
bb4204.i:		; preds = %bb4173.i
	%tmp4210.i = and i32 0, 32768		; <i32> [#uses=4]
	%tmp4212.i = and i32 %f_red.0.reg2mem.4.i, 2139095040		; <i32> [#uses=1]
	%tmp4214.i = and i32 %f_red.0.reg2mem.4.i, 8388607		; <i32> [#uses=1]
	br i1 false, label %bb4219.i, label %bb4241.i
bb4219.i:		; preds = %bb4204.i
	br i1 false, label %bb4272.i, label %bb4236.i
bb4236.i:		; preds = %bb4219.i
	br label %bb4272.i
bb4241.i:		; preds = %bb4204.i
	br i1 false, label %bb4246.i, label %bb4259.i
bb4246.i:		; preds = %bb4241.i
	%tmp4253.i = lshr i32 %tmp4214.i, 0		; <i32> [#uses=1]
	%tmp4253.masked.i = and i32 %tmp4253.i, 65535		; <i32> [#uses=1]
	br label %bb4272.i
bb4259.i:		; preds = %bb4241.i
	%tmp4261.i187 = add i32 %tmp4212.i, 134217728		; <i32> [#uses=1]
	%tmp4262.i188 = lshr i32 %tmp4261.i187, 13		; <i32> [#uses=1]
	%tmp4262.masked.i = and i32 %tmp4262.i188, 64512		; <i32> [#uses=1]
	%tmp42665693.masked.i = or i32 %tmp4262.masked.i, %tmp4210.i		; <i32> [#uses=1]
	br label %bb4272.i
bb4272.i:		; preds = %bb4259.i, %bb4246.i, %bb4236.i, %bb4219.i
	%tmp42665693.masked.pn.i = phi i32 [ %tmp42665693.masked.i, %bb4259.i ], [ %tmp4253.masked.i, %bb4246.i ], [ %tmp4210.i, %bb4236.i ], [ %tmp4210.i, %bb4219.i ]		; <i32> [#uses=1]
	%tmp4268.pn.i = phi i32 [ 0, %bb4259.i ], [ %tmp4210.i, %bb4246.i ], [ 31744, %bb4236.i ], [ 32767, %bb4219.i ]		; <i32> [#uses=1]
	%tmp100.0.i = or i32 %tmp4268.pn.i, %tmp42665693.masked.pn.i		; <i32> [#uses=0]
	%tmp4289.i = and i32 %f_green.0.reg2mem.2.i, 8388607		; <i32> [#uses=1]
	br i1 false, label %bb4294.i, label %bb4316.i
bb4294.i:		; preds = %bb4272.i
	br i1 false, label %bb4347.i, label %bb4311.i
bb4311.i:		; preds = %bb4294.i
	br label %bb4347.i
bb4316.i:		; preds = %bb4272.i
	br i1 false, label %bb4321.i, label %bb4334.i
bb4321.i:		; preds = %bb4316.i
	br label %bb4347.i
bb4334.i:		; preds = %bb4316.i
	%tmp4343.i = lshr i32 %tmp4289.i, 13		; <i32> [#uses=0]
	br label %bb4347.i
bb4347.i:		; preds = %bb4334.i, %bb4321.i, %bb4311.i, %bb4294.i
	%tmp4364.i190 = and i32 %f_blue.0.reg2mem.2.i, 8388607		; <i32> [#uses=1]
	br i1 false, label %bb4369.i192, label %bb4391.i
bb4369.i192:		; preds = %bb4347.i
	br i1 false, label %bb4422.i, label %bb4386.i
bb4386.i:		; preds = %bb4369.i192
	br label %bb4422.i
bb4391.i:		; preds = %bb4347.i
	br i1 false, label %bb4396.i, label %bb4409.i
bb4396.i:		; preds = %bb4391.i
	br label %bb4422.i
bb4409.i:		; preds = %bb4391.i
	%tmp4418.i = lshr i32 %tmp4364.i190, 13		; <i32> [#uses=0]
	br label %bb4422.i
bb4422.i:		; preds = %bb4409.i, %bb4396.i, %bb4386.i, %bb4369.i192
	%tmp4439.i194 = and i32 %f_alpha.1.reg2mem.1.i, 8388607		; <i32> [#uses=1]
	br i1 false, label %bb4444.i, label %bb4466.i
bb4444.i:		; preds = %bb4422.i
	br i1 false, label %bb4497.i, label %bb4461.i
bb4461.i:		; preds = %bb4444.i
	br label %glgScalarFloatConversion.exit
bb4466.i:		; preds = %bb4422.i
	br i1 false, label %bb4471.i, label %bb4484.i
bb4471.i:		; preds = %bb4466.i
	br label %glgScalarFloatConversion.exit
bb4484.i:		; preds = %bb4466.i
	%tmp4493.i = lshr i32 %tmp4439.i194, 13		; <i32> [#uses=0]
	br label %glgScalarFloatConversion.exit
bb4497.i:		; preds = %bb4444.i
	br label %glgScalarFloatConversion.exit
bb4517.i202:		; preds = %bb4173.i, %bb4173.i
	br label %glgScalarFloatConversion.exit
bb4575.i:		; preds = %bb4173.i
	br label %glgScalarFloatConversion.exit
bb4769.i:		; preds = %bb4173.i
	br label %glgScalarFloatConversion.exit
bb4950.i:		; preds = %bb4173.i
	br i1 false, label %bb4956.i, label %bb4993.i
bb4956.i:		; preds = %bb4950.i
	br label %glgScalarFloatConversion.exit
bb4993.i:		; preds = %bb4950.i
	br i1 false, label %bb4999.i, label %bb5036.i
bb4999.i:		; preds = %bb4993.i
	br label %glgScalarFloatConversion.exit
bb5036.i:		; preds = %bb4993.i
	br label %glgScalarFloatConversion.exit
UnifiedReturnBlock.i235:		; preds = %bb2883.i, %bb2872.i
	br label %glgScalarFloatConversion.exit
glgScalarFloatConversion.exit:		; preds = %UnifiedReturnBlock.i235, %bb5036.i, %bb4999.i, %bb4956.i, %bb4769.i, %bb4575.i, %bb4517.i202, %bb4497.i, %bb4484.i, %bb4471.i, %bb4461.i, %bb4178.i, %bb4145.i, %bb4117.i, %bb4083.i, %bb3941.i, %bb3794.i, %bb3749.i, %bb3729.i, %bb3716.i, %bb3703.i, %bb3693.i, %bb3490.i, %bb3475.i, %bb3429.i, %bb3376.i, %bb3357.i, %bb3337.i, %bb3324.i, %bb3311.i, %bb3301.i, %bb3258.i, %bb3011.i, %bb2972.i
	br label %bb18851.i
bb16697.i:		; preds = %loadColor_BGRA_UI8888R.exit
	br i1 false, label %bb17749.i, label %bb16700.i
bb16700.i:		; preds = %bb16697.i
	switch i32 0, label %bb16829.i [
		 i32 4, label %bb16705.i
		 i32 8, label %bb16743.i
		 i32 11, label %bb16795.i
	]
bb16705.i:		; preds = %bb16700.i
	switch i32 %df4081.0.i, label %bb17183.i [
		 i32 1, label %bb16710.i
		 i32 2, label %bb16721.i
		 i32 3, label %bb16732.i
	]
bb16710.i:		; preds = %bb16705.i
	br label %bb17195.i
bb16721.i:		; preds = %bb16705.i
	br label %bb17195.i
bb16732.i:		; preds = %bb16705.i
	br label %bb17195.i
bb16743.i:		; preds = %bb16700.i
	switch i32 0, label %bb16759.i [
		 i32 4, label %bb16755.i
		 i32 11, label %bb16755.i
	]
bb16755.i:		; preds = %bb16743.i, %bb16743.i
	br label %bb17195.i
bb16759.i:		; preds = %bb16743.i
	switch i32 %df4081.0.i, label %bb17183.i [
		 i32 1, label %bb16764.i
		 i32 2, label %bb16775.i
		 i32 3, label %bb16786.i
	]
bb16764.i:		; preds = %bb16759.i
	br label %bb17195.i
bb16775.i:		; preds = %bb16759.i
	br label %bb17195.i
bb16786.i:		; preds = %bb16759.i
	br label %bb17195.i
bb16795.i:		; preds = %bb16700.i
	switch i32 0, label %bb17183.i [
		 i32 4, label %bb16807.i
		 i32 8, label %bb16807.i
		 i32 3, label %bb16823.i
	]
bb16807.i:		; preds = %bb16795.i, %bb16795.i
	br label %bb17195.i
bb16823.i:		; preds = %bb16795.i
	br label %bb17195.i
bb16829.i:		; preds = %bb16700.i
	switch i32 %sf4083.0.i, label %bb17183.i [
		 i32 10, label %bb16834.i
		 i32 0, label %bb16892.i
		 i32 1, label %bb16953.i
		 i32 2, label %bb17037.i
		 i32 3, label %bb17121.i
	]
bb16834.i:		; preds = %bb16829.i
	switch i32 0, label %bb16878.i [
		 i32 4, label %bb16839.i
		 i32 8, label %bb16858.i
		 i32 11, label %bb16874.i
	]
bb16839.i:		; preds = %bb16834.i
	br label %bb17195.i
bb16858.i:		; preds = %bb16834.i
	br label %bb17195.i
bb16874.i:		; preds = %bb16834.i
	br label %bb17195.i
bb16878.i:		; preds = %bb16834.i
	br i1 false, label %bb16883.i, label %bb17183.i
bb16883.i:		; preds = %bb16878.i
	br label %bb17195.i
bb16892.i:		; preds = %bb16829.i
	switch i32 0, label %bb16930.i [
		 i32 4, label %bb16897.i
		 i32 8, label %bb16913.i
		 i32 11, label %bb16926.i
	]
bb16897.i:		; preds = %bb16892.i
	br label %bb17195.i
bb16913.i:		; preds = %bb16892.i
	br label %bb17195.i
bb16926.i:		; preds = %bb16892.i
	br label %bb17195.i
bb16930.i:		; preds = %bb16892.i
	br i1 false, label %bb16936.i, label %bb16939.i
bb16936.i:		; preds = %bb16930.i
	br label %bb17195.i
bb16939.i:		; preds = %bb16930.i
	br i1 false, label %bb16944.i, label %bb17183.i
bb16944.i:		; preds = %bb16939.i
	br label %bb17195.i
bb16953.i:		; preds = %bb16829.i
	switch i32 0, label %bb17003.i [
		 i32 4, label %bb16958.i
		 i32 8, label %bb16979.i
		 i32 11, label %bb16997.i
	]
bb16958.i:		; preds = %bb16953.i
	br label %bb17195.i
bb16979.i:		; preds = %bb16953.i
	br label %bb17195.i
bb16997.i:		; preds = %bb16953.i
	br label %bb17195.i
bb17003.i:		; preds = %bb16953.i
	switch i32 %df4081.0.i, label %bb17183.i [
		 i32 0, label %bb17020.i
		 i32 2, label %bb17020.i
		 i32 10, label %bb17020.i
		 i32 3, label %bb17028.i
	]
bb17020.i:		; preds = %bb17003.i, %bb17003.i, %bb17003.i
	br label %bb17195.i
bb17028.i:		; preds = %bb17003.i
	br label %bb17195.i
bb17037.i:		; preds = %bb16829.i
	switch i32 0, label %bb17087.i [
		 i32 4, label %bb17042.i
		 i32 8, label %bb17063.i
		 i32 11, label %bb17081.i
	]
bb17042.i:		; preds = %bb17037.i
	br label %bb17195.i
bb17063.i:		; preds = %bb17037.i
	br label %bb17195.i
bb17081.i:		; preds = %bb17037.i
	br label %bb17195.i
bb17087.i:		; preds = %bb17037.i
	switch i32 %df4081.0.i, label %bb17183.i [
		 i32 0, label %bb17104.i
		 i32 1, label %bb17104.i
		 i32 10, label %bb17104.i
		 i32 3, label %bb17112.i
	]
bb17104.i:		; preds = %bb17087.i, %bb17087.i, %bb17087.i
	br label %bb17195.i
bb17112.i:		; preds = %bb17087.i
	br label %bb17195.i
bb17121.i:		; preds = %bb16829.i
	switch i32 0, label %bb17183.i [
		 i32 4, label %bb17126.i
		 i32 8, label %bb17149.i
		 i32 11, label %bb17167.i
		 i32 10, label %bb17180.i
	]
bb17126.i:		; preds = %bb17121.i
	br label %bb17195.i
bb17149.i:		; preds = %bb17121.i
	br label %bb17195.i
bb17167.i:		; preds = %bb17121.i
	br label %bb17195.i
bb17180.i:		; preds = %bb17121.i
	br label %bb17183.i
bb17183.i:		; preds = %bb17180.i, %bb17121.i, %bb17087.i, %bb17003.i, %bb16939.i, %bb16878.i, %bb16829.i, %bb16795.i, %bb16759.i, %bb16705.i
	br label %bb17195.i
bb17195.i:		; preds = %bb17183.i, %bb17167.i, %bb17149.i, %bb17126.i, %bb17112.i, %bb17104.i, %bb17081.i, %bb17063.i, %bb17042.i, %bb17028.i, %bb17020.i, %bb16997.i, %bb16979.i, %bb16958.i, %bb16944.i, %bb16936.i, %bb16926.i, %bb16913.i, %bb16897.i, %bb16883.i, %bb16874.i, %bb16858.i, %bb16839.i, %bb16823.i, %bb16807.i, %bb16786.i, %bb16775.i, %bb16764.i, %bb16755.i, %bb16732.i, %bb16721.i, %bb16710.i
	br i1 false, label %bb18845.i, label %bb17225.i
bb17225.i:		; preds = %bb17195.i
	switch i32 %dt4080.0.i, label %bb17677.i [
		 i32 4, label %bb17227.i
		 i32 8, label %bb17259.i
		 i32 9, label %bb17309.i
		 i32 10, label %bb17359.i
		 i32 11, label %bb17359.i
		 i32 14, label %bb17409.i
		 i32 15, label %bb17474.i
		 i32 18, label %bb17539.i
		 i32 19, label %bb17604.i
		 i32 0, label %bb17680.i
		 i32 1, label %bb17672.i
		 i32 2, label %bb17673.i
		 i32 3, label %bb17674.i
		 i32 5, label %bb17675.i
		 i32 12, label %bb17676.i
		 i32 13, label %bb17676.i
		 i32 16, label %bb17680.i
		 i32 17, label %bb17680.i
	]
bb17227.i:		; preds = %bb17225.i
	br i1 false, label %bb18845.i, label %bb17230.i
bb17230.i:		; preds = %bb17227.i
	br label %bb18851.i
bb17259.i:		; preds = %bb17225.i
	br i1 false, label %bb17284.i, label %bb17262.i
bb17262.i:		; preds = %bb17259.i
	br label %bb17284.i
bb17284.i:		; preds = %bb17262.i, %bb17259.i
	br label %bb18851.i
bb17309.i:		; preds = %bb17225.i
	br i1 false, label %bb17334.i, label %bb17312.i
bb17312.i:		; preds = %bb17309.i
	br label %bb17334.i
bb17334.i:		; preds = %bb17312.i, %bb17309.i
	br label %bb18851.i
bb17359.i:		; preds = %bb17225.i, %bb17225.i
	br i1 false, label %bb17384.i, label %bb17362.i
bb17362.i:		; preds = %bb17359.i
	br label %bb17384.i
bb17384.i:		; preds = %bb17362.i, %bb17359.i
	br label %bb18851.i
bb17409.i:		; preds = %bb17225.i
	br i1 false, label %bb17441.i, label %bb17412.i
bb17412.i:		; preds = %bb17409.i
	br label %bb17441.i
bb17441.i:		; preds = %bb17412.i, %bb17409.i
	br label %bb18851.i
bb17474.i:		; preds = %bb17225.i
	br i1 false, label %bb17506.i, label %bb17477.i
bb17477.i:		; preds = %bb17474.i
	br label %bb17506.i
bb17506.i:		; preds = %bb17477.i, %bb17474.i
	br label %bb18851.i
bb17539.i:		; preds = %bb17225.i
	br i1 false, label %bb17571.i, label %bb17542.i
bb17542.i:		; preds = %bb17539.i
	br label %bb17571.i
bb17571.i:		; preds = %bb17542.i, %bb17539.i
	br label %bb18851.i
bb17604.i:		; preds = %bb17225.i
	br i1 false, label %bb17636.i, label %bb17607.i
bb17607.i:		; preds = %bb17604.i
	br label %bb17636.i
bb17636.i:		; preds = %bb17607.i, %bb17604.i
	br label %bb18851.i
bb17672.i:		; preds = %bb17225.i
	br i1 false, label %bb17716.i, label %bb17683.i
bb17673.i:		; preds = %bb17225.i
	br i1 false, label %bb17716.i, label %bb17683.i
bb17674.i:		; preds = %bb17225.i
	br i1 false, label %bb17716.i, label %bb17683.i
bb17675.i:		; preds = %bb17225.i
	br i1 false, label %bb17716.i, label %bb17683.i
bb17676.i:		; preds = %bb17225.i, %bb17225.i
	br i1 false, label %bb17716.i, label %bb17683.i
bb17677.i:		; preds = %bb17225.i
	unreachable
bb17680.i:		; preds = %bb17225.i, %bb17225.i, %bb17225.i
	br i1 false, label %bb17716.i, label %bb17683.i
bb17683.i:		; preds = %bb17680.i, %bb17676.i, %bb17675.i, %bb17674.i, %bb17673.i, %bb17672.i
	br label %bb17716.i
bb17716.i:		; preds = %bb17683.i, %bb17680.i, %bb17676.i, %bb17675.i, %bb17674.i, %bb17673.i, %bb17672.i
	br label %bb18851.i
bb17749.i:		; preds = %bb16697.i
	br i1 false, label %bb17757.i, label %bb17903.i
bb17757.i:		; preds = %bb17749.i
	switch i32 0, label %bb17903.i [
		 i32 0, label %bb17759.i
		 i32 1, label %bb17853.i
		 i32 2, label %bb17853.i
	]
bb17759.i:		; preds = %bb17757.i
	br i1 false, label %bb17764.i, label %bb17772.i
bb17764.i:		; preds = %bb17759.i
	br label %bb18032.i
bb17772.i:		; preds = %bb17759.i
	switch i32 %sf4083.0.i, label %bb17798.i [
		 i32 1, label %bb17777.i
		 i32 2, label %bb17790.i
	]
bb17777.i:		; preds = %bb17772.i
	switch i32 0, label %bb18032.i [
		 i32 4, label %bb17818.i
		 i32 8, label %bb17818.i
		 i32 11, label %bb17845.i
	]
bb17790.i:		; preds = %bb17772.i
	switch i32 0, label %bb18032.i [
		 i32 4, label %bb17818.i
		 i32 8, label %bb17818.i
		 i32 11, label %bb17845.i
	]
bb17798.i:		; preds = %bb17772.i
	switch i32 0, label %bb18032.i [
		 i32 4, label %bb17818.i
		 i32 8, label %bb17818.i
		 i32 11, label %bb17845.i
	]
bb17818.i:		; preds = %bb17798.i, %bb17798.i, %bb17790.i, %bb17790.i, %bb17777.i, %bb17777.i
	switch i32 0, label %bb18032.i [
		 i32 4, label %bb17845.i
		 i32 11, label %bb17845.i
		 i32 8, label %bb17946.i
	]
bb17845.i:		; preds = %bb17818.i, %bb17818.i, %bb17798.i, %bb17790.i, %bb17777.i
	switch i32 0, label %bb18032.i [
		 i32 4, label %bb17908.i
		 i32 8, label %bb17946.i
		 i32 11, label %bb17998.i
	]
bb17853.i:		; preds = %bb17757.i, %bb17757.i
	br i1 false, label %bb17890.i, label %bb17903.i
bb17890.i:		; preds = %bb17853.i
	br label %bb17903.i
bb17903.i:		; preds = %bb17890.i, %bb17853.i, %bb17757.i, %bb17749.i
	switch i32 0, label %bb18032.i [
		 i32 4, label %bb17908.i
		 i32 8, label %bb17946.i
		 i32 11, label %bb17998.i
	]
bb17908.i:		; preds = %bb17903.i, %bb17845.i
	switch i32 %df4081.0.i, label %bb18386.i [
		 i32 1, label %bb17913.i
		 i32 2, label %bb17924.i
		 i32 3, label %bb17935.i
	]
bb17913.i:		; preds = %bb17908.i
	br label %bb18398.i
bb17924.i:		; preds = %bb17908.i
	br label %bb18398.i
bb17935.i:		; preds = %bb17908.i
	br label %bb18398.i
bb17946.i:		; preds = %bb17903.i, %bb17845.i, %bb17818.i
	switch i32 0, label %bb17962.i [
		 i32 4, label %bb17958.i
		 i32 11, label %bb17958.i
	]
bb17958.i:		; preds = %bb17946.i, %bb17946.i
	br label %bb18398.i
bb17962.i:		; preds = %bb17946.i
	switch i32 %df4081.0.i, label %bb18386.i [
		 i32 1, label %bb17967.i
		 i32 2, label %bb17978.i
		 i32 3, label %bb17989.i
	]
bb17967.i:		; preds = %bb17962.i
	br label %bb18398.i
bb17978.i:		; preds = %bb17962.i
	br label %bb18398.i
bb17989.i:		; preds = %bb17962.i
	br label %bb18398.i
bb17998.i:		; preds = %bb17903.i, %bb17845.i
	switch i32 0, label %bb18386.i [
		 i32 4, label %bb18010.i
		 i32 8, label %bb18010.i
		 i32 3, label %bb18026.i
	]
bb18010.i:		; preds = %bb17998.i, %bb17998.i
	br label %bb18398.i
bb18026.i:		; preds = %bb17998.i
	br label %bb18398.i
bb18032.i:		; preds = %bb17903.i, %bb17845.i, %bb17818.i, %bb17798.i, %bb17790.i, %bb17777.i, %bb17764.i
	switch i32 %sf4083.0.i, label %bb18386.i [
		 i32 10, label %bb18037.i
		 i32 0, label %bb18095.i
		 i32 1, label %bb18156.i
		 i32 2, label %bb18240.i
		 i32 3, label %bb18324.i
	]
bb18037.i:		; preds = %bb18032.i
	switch i32 0, label %bb18081.i [
		 i32 4, label %bb18042.i
		 i32 8, label %bb18061.i
		 i32 11, label %bb18077.i
	]
bb18042.i:		; preds = %bb18037.i
	br label %bb18398.i
bb18061.i:		; preds = %bb18037.i
	br label %bb18398.i
bb18077.i:		; preds = %bb18037.i
	br label %bb18398.i
bb18081.i:		; preds = %bb18037.i
	br i1 false, label %bb18086.i, label %bb18386.i
bb18086.i:		; preds = %bb18081.i
	br label %bb18398.i
bb18095.i:		; preds = %bb18032.i
	switch i32 0, label %bb18133.i [
		 i32 4, label %bb18100.i
		 i32 8, label %bb18116.i
		 i32 11, label %bb18129.i
	]
bb18100.i:		; preds = %bb18095.i
	br label %bb18398.i
bb18116.i:		; preds = %bb18095.i
	br label %bb18398.i
bb18129.i:		; preds = %bb18095.i
	br label %bb18398.i
bb18133.i:		; preds = %bb18095.i
	br i1 false, label %bb18139.i, label %bb18142.i
bb18139.i:		; preds = %bb18133.i
	br label %bb18398.i
bb18142.i:		; preds = %bb18133.i
	br i1 false, label %bb18147.i, label %bb18386.i
bb18147.i:		; preds = %bb18142.i
	br label %bb18398.i
bb18156.i:		; preds = %bb18032.i
	switch i32 0, label %bb18206.i [
		 i32 4, label %bb18161.i
		 i32 8, label %bb18182.i
		 i32 11, label %bb18200.i
	]
bb18161.i:		; preds = %bb18156.i
	br label %bb18398.i
bb18182.i:		; preds = %bb18156.i
	br label %bb18398.i
bb18200.i:		; preds = %bb18156.i
	br label %bb18398.i
bb18206.i:		; preds = %bb18156.i
	switch i32 %df4081.0.i, label %bb18386.i [
		 i32 0, label %bb18223.i
		 i32 2, label %bb18223.i
		 i32 10, label %bb18223.i
		 i32 3, label %bb18231.i
	]
bb18223.i:		; preds = %bb18206.i, %bb18206.i, %bb18206.i
	br label %bb18398.i
bb18231.i:		; preds = %bb18206.i
	br label %bb18398.i
bb18240.i:		; preds = %bb18032.i
	switch i32 0, label %bb18290.i [
		 i32 4, label %bb18245.i
		 i32 8, label %bb18266.i
		 i32 11, label %bb18284.i
	]
bb18245.i:		; preds = %bb18240.i
	br label %bb18398.i
bb18266.i:		; preds = %bb18240.i
	br label %bb18398.i
bb18284.i:		; preds = %bb18240.i
	br label %bb18398.i
bb18290.i:		; preds = %bb18240.i
	switch i32 %df4081.0.i, label %bb18386.i [
		 i32 0, label %bb18307.i
		 i32 1, label %bb18307.i
		 i32 10, label %bb18307.i
		 i32 3, label %bb18315.i
	]
bb18307.i:		; preds = %bb18290.i, %bb18290.i, %bb18290.i
	br label %bb18398.i
bb18315.i:		; preds = %bb18290.i
	br label %bb18398.i
bb18324.i:		; preds = %bb18032.i
	switch i32 0, label %bb18386.i [
		 i32 4, label %bb18329.i
		 i32 8, label %bb18352.i
		 i32 11, label %bb18370.i
		 i32 10, label %bb18383.i
	]
bb18329.i:		; preds = %bb18324.i
	br label %bb18398.i
bb18352.i:		; preds = %bb18324.i
	br label %bb18398.i
bb18370.i:		; preds = %bb18324.i
	br label %bb18398.i
bb18383.i:		; preds = %bb18324.i
	br label %bb18386.i
bb18386.i:		; preds = %bb18383.i, %bb18324.i, %bb18290.i, %bb18206.i, %bb18142.i, %bb18081.i, %bb18032.i, %bb17998.i, %bb17962.i, %bb17908.i
	br label %bb18398.i
bb18398.i:		; preds = %bb18386.i, %bb18370.i, %bb18352.i, %bb18329.i, %bb18315.i, %bb18307.i, %bb18284.i, %bb18266.i, %bb18245.i, %bb18231.i, %bb18223.i, %bb18200.i, %bb18182.i, %bb18161.i, %bb18147.i, %bb18139.i, %bb18129.i, %bb18116.i, %bb18100.i, %bb18086.i, %bb18077.i, %bb18061.i, %bb18042.i, %bb18026.i, %bb18010.i, %bb17989.i, %bb17978.i, %bb17967.i, %bb17958.i, %bb17935.i, %bb17924.i, %bb17913.i
	br i1 false, label %bb18589.i, label %bb18431.i
bb18431.i:		; preds = %bb18398.i
	switch i32 0, label %bb18589.i [
		 i32 0, label %bb18433.i
		 i32 1, label %bb18487.i
		 i32 2, label %bb18487.i
	]
bb18433.i:		; preds = %bb18431.i
	switch i32 0, label %bb18589.i [
		 i32 4, label %bb18452.i
		 i32 8, label %bb18452.i
		 i32 11, label %bb18479.i
	]
bb18452.i:		; preds = %bb18433.i, %bb18433.i
	switch i32 0, label %bb18589.i [
		 i32 4, label %bb18479.i
		 i32 11, label %bb18479.i
	]
bb18479.i:		; preds = %bb18452.i, %bb18452.i, %bb18433.i
	br i1 false, label %bb18845.i, label %bb18592.i
bb18487.i:		; preds = %bb18431.i, %bb18431.i
	br i1 false, label %bb18492.i, label %bb18521.i
bb18492.i:		; preds = %bb18487.i
	br i1 false, label %bb18508.i, label %bb18529.i
bb18508.i:		; preds = %bb18492.i
	switch i32 0, label %bb18589.i [
		 i32 4, label %bb18541.i
		 i32 8, label %bb18541.i
	]
bb18521.i:		; preds = %bb18487.i
	br label %bb18529.i
bb18529.i:		; preds = %bb18521.i, %bb18492.i
	switch i32 0, label %bb18589.i [
		 i32 4, label %bb18541.i
		 i32 8, label %bb18541.i
	]
bb18541.i:		; preds = %bb18529.i, %bb18529.i, %bb18508.i, %bb18508.i
	br i1 false, label %bb18560.i, label %bb18589.i
bb18560.i:		; preds = %bb18541.i
	br i1 false, label %bb18576.i, label %bb18589.i
bb18576.i:		; preds = %bb18560.i
	br label %bb18589.i
bb18589.i:		; preds = %bb18576.i, %bb18560.i, %bb18541.i, %bb18529.i, %bb18508.i, %bb18452.i, %bb18433.i, %bb18431.i, %bb18398.i
	br i1 false, label %bb18845.i, label %bb18592.i
bb18592.i:		; preds = %bb18589.i, %bb18479.i
	switch i32 %dt4080.0.i, label %bb18809.i [
		 i32 4, label %bb18845.i
		 i32 8, label %bb18594.i
		 i32 9, label %bb18619.i
		 i32 10, label %bb18644.i
		 i32 11, label %bb18644.i
		 i32 14, label %bb18669.i
		 i32 15, label %bb18702.i
		 i32 18, label %bb18735.i
		 i32 19, label %bb18768.i
		 i32 0, label %bb18812.i
		 i32 1, label %bb18804.i
		 i32 2, label %bb18805.i
		 i32 3, label %bb18806.i
		 i32 5, label %bb18807.i
		 i32 12, label %bb18808.i
		 i32 13, label %bb18808.i
		 i32 16, label %bb18812.i
		 i32 17, label %bb18812.i
	]
bb18594.i:		; preds = %bb18592.i
	br label %bb18851.i
bb18619.i:		; preds = %bb18592.i
	br label %bb18851.i
bb18644.i:		; preds = %bb18592.i, %bb18592.i
	br label %bb18851.i
bb18669.i:		; preds = %bb18592.i
	br label %bb18851.i
bb18702.i:		; preds = %bb18592.i
	br label %bb18851.i
bb18735.i:		; preds = %bb18592.i
	br label %bb18851.i
bb18768.i:		; preds = %bb18592.i
	br label %bb18851.i
bb18804.i:		; preds = %bb18592.i
	br label %bb18812.i
bb18805.i:		; preds = %bb18592.i
	br label %bb18812.i
bb18806.i:		; preds = %bb18592.i
	br label %bb18812.i
bb18807.i:		; preds = %bb18592.i
	br label %bb18812.i
bb18808.i:		; preds = %bb18592.i, %bb18592.i
	br label %bb18812.i
bb18809.i:		; preds = %bb18592.i
	unreachable
bb18812.i:		; preds = %bb18808.i, %bb18807.i, %bb18806.i, %bb18805.i, %bb18804.i, %bb18592.i, %bb18592.i, %bb18592.i
	br label %bb18845.i
bb18845.i:		; preds = %bb18812.i, %bb18592.i, %bb18589.i, %bb18479.i, %bb17227.i, %bb17195.i
	br label %bb18851.i
bb18851.i:		; preds = %bb18845.i, %bb18768.i, %bb18735.i, %bb18702.i, %bb18669.i, %bb18644.i, %bb18619.i, %bb18594.i, %bb17716.i, %bb17636.i, %bb17571.i, %bb17506.i, %bb17441.i, %bb17384.i, %bb17334.i, %bb17284.i, %bb17230.i, %glgScalarFloatConversion.exit
	br label %storeColor_RGB_UI.exit
storeColor_RGB_UI.exit:		; preds = %bb18851.i
	br i1 false, label %bb19786.i, label %bb16650.i
bb19786.i:		; preds = %storeColor_RGB_UI.exit
	br label %bb19808.i
bb19808.i:		; preds = %bb19786.i
	br i1 false, label %bb19818.i, label %bb5276.i
bb19818.i:		; preds = %bb19808.i
	br i1 false, label %bb19840.i, label %bb19821.i
bb19821.i:		; preds = %bb19818.i
	br label %bb19840.i
bb19840.i:		; preds = %bb19821.i, %bb19818.i
	br i1 false, label %UnifiedReturnBlock.i, label %bb19843.i
bb19843.i:		; preds = %bb19840.i
	br label %t.exit
UnifiedReturnBlock.i:		; preds = %bb19840.i, %bb4501.i
	br label %t.exit
t.exit:		; preds = %UnifiedReturnBlock.i, %bb19843.i, %bb4517.i, %bb4354.i
	ret void
}
