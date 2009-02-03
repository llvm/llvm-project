; RUN: llvm-as < %s | opt -loop-index-split -disable-output
	%struct.RExC_state_t = type { i32, i8*, %struct.regexp*, i8*, i8*, i8*, i32, %struct.regnode*, %struct.regnode*, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.SV = type { i8*, i32, i32 }
	%struct.reg_data = type { i32, i8*, [1 x i8*] }
	%struct.reg_substr_data = type { [3 x %struct.reg_substr_datum] }
	%struct.reg_substr_datum = type { i32, i32, %struct.SV*, %struct.SV* }
	%struct.regexp = type { i32*, i32*, %struct.regnode*, %struct.reg_substr_data*, i8*, %struct.reg_data*, i8*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, [1 x %struct.regnode] }
	%struct.regnode = type { i8, i8, i16 }

define fastcc %struct.regnode* @S_regclass(%struct.RExC_state_t* %pRExC_state) nounwind {
entry:
	br label %bb439

bb439:		; preds = %bb444, %entry
	%value23.16.reg2mem.0 = phi i32 [ %3, %bb444 ], [ 0, %entry ]		; <i32> [#uses=3]
	%0 = icmp ugt i32 %value23.16.reg2mem.0, 31		; <i1> [#uses=1]
	%1 = icmp ne i32 %value23.16.reg2mem.0, 127		; <i1> [#uses=1]
	%2 = and i1 %0, %1		; <i1> [#uses=1]
	br i1 %2, label %bb443, label %bb444

bb443:		; preds = %bb439
	br label %bb444

bb444:		; preds = %bb443, %bb439
	%3 = add i32 %value23.16.reg2mem.0, 1		; <i32> [#uses=2]
	%4 = icmp ugt i32 %3, 255		; <i1> [#uses=1]
	br i1 %4, label %bb675, label %bb439

bb675:		; preds = %bb444
	unreachable
}
