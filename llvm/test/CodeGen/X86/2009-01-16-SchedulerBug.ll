; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin
; rdar://6501631

	%CF = type { %Register }
	%XXV = type { i32 (...)** }
	%Register = type { %"struct.XXC::BCFs", i32 }
	%"struct.XXC::BCFs" = type { i32 }

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind

define fastcc %XXV* @bar(%CF* %call_frame, %XXV** %exception) nounwind {
prologue:
	%param_x = load %XXV** null		; <%XXV*> [#uses=1]
	%unique_1.i = ptrtoint %XXV* %param_x to i1		; <i1> [#uses=1]
	br i1 %unique_1.i, label %NextVerify42, label %FailedVerify

NextVerify42:		; preds = %prologue
	%param_y = load %XXV** null		; <%XXV*> [#uses=1]
	%unique_1.i58 = ptrtoint %XXV* %param_y to i1		; <i1> [#uses=1]
	br i1 %unique_1.i58, label %function_setup.cont, label %FailedVerify

function_setup.cont:		; preds = %NextVerify42
	br i1 false, label %label13, label %label

label:		; preds = %function_setup.cont
	%has_exn = icmp eq %XXV* null, null		; <i1> [#uses=1]
	br i1 %has_exn, label %kjsNumberLiteral.exit, label %handle_exception

kjsNumberLiteral.exit:		; preds = %label
	%0 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 0, i32 0)		; <{ i32, i1 }> [#uses=2]
	%intAdd = extractvalue { i32, i1 } %0, 0		; <i32> [#uses=2]
	%intAddOverflow = extractvalue { i32, i1 } %0, 1		; <i1> [#uses=1]
	%toint56 = ashr i32 %intAdd, 1		; <i32> [#uses=1]
	%toFP57 = sitofp i32 %toint56 to double		; <double> [#uses=1]
	br i1 %intAddOverflow, label %rematerializeAdd, label %label13

label13:		; preds = %kjsNumberLiteral.exit, %function_setup.cont
	%var_lr1.0 = phi double [ %toFP57, %kjsNumberLiteral.exit ], [ 0.000000e+00, %function_setup.cont ]		; <double> [#uses=0]
	unreachable

FailedVerify:		; preds = %NextVerify42, %prologue
	ret %XXV* null

rematerializeAdd:		; preds = %kjsNumberLiteral.exit
	%rematerializedInt = sub i32 %intAdd, 0		; <i32> [#uses=0]
	ret %XXV* null

handle_exception:		; preds = %label
	ret %XXV* undef
}
