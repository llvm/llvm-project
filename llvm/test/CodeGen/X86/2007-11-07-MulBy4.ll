; RUN: llc < %s -march=x86 | not grep imul

	%struct.eebb = type { %struct.eebb*, i16* }
	%struct.hf = type { %struct.hf*, i16*, i8*, i32, i32, %struct.eebb*, i32, i32, i8*, i8*, i8*, i8*, i16*, i8*, i16*, %struct.ri, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [30 x i32], %struct.eebb, i32, i8* }
	%struct.foo_data = type { i32, i32, i32, i32*, i32, i32, i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i16*, i16*, i16*, i16*, i32, i32, i32, %struct.ri*, i8*, %struct.hf* }
	%struct.ri = type { %struct.ri*, i32, i8*, i16*, i32*, i32 }

define fastcc i32 @foo(i16* %eptr, i8* %ecode, %struct.foo_data* %md, i32 %ims) {
entry:
	%tmp36 = load i32, i32* null, align 4		; <i32> [#uses=1]
	%tmp37 = icmp ult i32 0, %tmp36		; <i1> [#uses=1]
	br i1 %tmp37, label %cond_next79, label %cond_true

cond_true:		; preds = %entry
	ret i32 0

cond_next79:		; preds = %entry
	%tmp85 = load i32, i32* null, align 4		; <i32> [#uses=1]
	%tmp86 = icmp ult i32 0, %tmp85		; <i1> [#uses=1]
	br i1 %tmp86, label %cond_next130, label %cond_true89

cond_true89:		; preds = %cond_next79
	ret i32 0

cond_next130:		; preds = %cond_next79
	%tmp173 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp173, label %cond_next201, label %cond_true176

cond_true176:		; preds = %cond_next130
	ret i32 0

cond_next201:		; preds = %cond_next130
	switch i32 0, label %bb19955 [
		 i32 0, label %bb1266
		 i32 1, label %bb5018
		 i32 2, label %bb5075
		 i32 3, label %cond_true5534
		 i32 4, label %cond_true5534
		 i32 5, label %bb6039
		 i32 6, label %bb6181
		 i32 7, label %bb6323
		 i32 8, label %bb6463
		 i32 9, label %bb6605
		 i32 10, label %bb6746
		 i32 11, label %cond_next5871
		 i32 16, label %bb5452
		 i32 17, label %bb5395
		 i32 19, label %bb4883
		 i32 20, label %bb5136
		 i32 23, label %bb12899
		 i32 64, label %bb2162
		 i32 69, label %bb1447
		 i32 70, label %bb1737
		 i32 71, label %bb1447
		 i32 72, label %bb1737
		 i32 73, label %cond_true1984
		 i32 75, label %bb740
		 i32 80, label %bb552
	]

bb552:		; preds = %cond_next201
	ret i32 0

bb740:		; preds = %cond_next201
	ret i32 0

bb1266:		; preds = %cond_next201
	ret i32 0

bb1447:		; preds = %cond_next201, %cond_next201
	ret i32 0

bb1737:		; preds = %cond_next201, %cond_next201
	ret i32 0

cond_true1984:		; preds = %cond_next201
	ret i32 0

bb2162:		; preds = %cond_next201
	ret i32 0

bb4883:		; preds = %cond_next201
	ret i32 0

bb5018:		; preds = %cond_next201
	ret i32 0

bb5075:		; preds = %cond_next201
	ret i32 0

bb5136:		; preds = %cond_next201
	ret i32 0

bb5395:		; preds = %cond_next201
	ret i32 0

bb5452:		; preds = %cond_next201
	ret i32 0

cond_true5534:		; preds = %cond_next201, %cond_next201
	ret i32 0

cond_next5871:		; preds = %cond_next201
	ret i32 0

bb6039:		; preds = %cond_next201
	ret i32 0

bb6181:		; preds = %cond_next201
	ret i32 0

bb6323:		; preds = %cond_next201
	ret i32 0

bb6463:		; preds = %cond_next201
	ret i32 0

bb6605:		; preds = %cond_next201
	ret i32 0

bb6746:		; preds = %cond_next201
	ret i32 0

bb12899:		; preds = %cond_next201
	ret i32 0

bb19955:		; preds = %cond_next201
	ret i32 0
}
