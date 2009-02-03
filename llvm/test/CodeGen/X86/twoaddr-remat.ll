; RUN: llvm-as < %s | llc -march=x86 | grep 59796 | count 3

	%Args = type %Value*
	%Exec = type opaque*
	%Identifier = type opaque*
	%JSFunction = type %Value (%Exec, %Scope, %Value, %Args)
	%PropertyNameArray = type opaque*
	%Scope = type opaque*
	%Value = type opaque*

declare i1 @X1(%Exec) readonly 

declare %Value @X2(%Exec)

declare i32 @X3(%Exec, %Value)

declare %Value @X4(i32) readnone 

define internal %Value @fast3bitlookup(%Exec %exec, %Scope %scope, %Value %this, %Args %args) nounwind {
prologue:
	%eh_check = tail call i1 @X1( %Exec %exec ) readonly 		; <i1> [#uses=1]
	br i1 %eh_check, label %exception, label %no_exception

exception:		; preds = %no_exception, %prologue
	%rethrow_result = tail call %Value @X2( %Exec %exec )		; <%Value> [#uses=1]
	ret %Value %rethrow_result

no_exception:		; preds = %prologue
	%args_intptr = bitcast %Args %args to i32*		; <i32*> [#uses=1]
	%argc_val = load i32* %args_intptr		; <i32> [#uses=1]
	%cmpParamArgc = icmp sgt i32 %argc_val, 0		; <i1> [#uses=1]
	%arg_ptr = getelementptr %Args %args, i32 1		; <%Args> [#uses=1]
	%arg_val = load %Args %arg_ptr		; <%Value> [#uses=1]
	%ext_arg_val = select i1 %cmpParamArgc, %Value %arg_val, %Value inttoptr (i32 5 to %Value)		; <%Value> [#uses=1]
	%toInt325 = tail call i32 @X3( %Exec %exec, %Value %ext_arg_val )		; <i32> [#uses=3]
	%eh_check6 = tail call i1 @X1( %Exec %exec ) readonly 		; <i1> [#uses=1]
	br i1 %eh_check6, label %exception, label %no_exception7

no_exception7:		; preds = %no_exception
	%shl_tmp_result = shl i32 %toInt325, 1		; <i32> [#uses=1]
	%rhs_masked13 = and i32 %shl_tmp_result, 14		; <i32> [#uses=1]
	%ashr_tmp_result = lshr i32 59796, %rhs_masked13		; <i32> [#uses=1]
	%and_tmp_result15 = and i32 %ashr_tmp_result, 3		; <i32> [#uses=1]
	%ashr_tmp_result3283 = lshr i32 %toInt325, 2		; <i32> [#uses=1]
	%rhs_masked38 = and i32 %ashr_tmp_result3283, 14		; <i32> [#uses=1]
	%ashr_tmp_result39 = lshr i32 59796, %rhs_masked38		; <i32> [#uses=1]
	%and_tmp_result41 = and i32 %ashr_tmp_result39, 3		; <i32> [#uses=1]
	%addconv = add i32 %and_tmp_result15, %and_tmp_result41		; <i32> [#uses=1]
	%ashr_tmp_result6181 = lshr i32 %toInt325, 5		; <i32> [#uses=1]
	%rhs_masked67 = and i32 %ashr_tmp_result6181, 6		; <i32> [#uses=1]
	%ashr_tmp_result68 = lshr i32 59796, %rhs_masked67		; <i32> [#uses=1]
	%and_tmp_result70 = and i32 %ashr_tmp_result68, 3		; <i32> [#uses=1]
	%addconv82 = add i32 %addconv, %and_tmp_result70		; <i32> [#uses=3]
	%rangetmp = add i32 %addconv82, 536870912		; <i32> [#uses=1]
	%rangecmp = icmp ult i32 %rangetmp, 1073741824		; <i1> [#uses=1]
	br i1 %rangecmp, label %NumberLiteralIntFast, label %NumberLiteralIntSlow

NumberLiteralIntFast:		; preds = %no_exception7
	%imm_shift = shl i32 %addconv82, 2		; <i32> [#uses=1]
	%imm_or = or i32 %imm_shift, 3		; <i32> [#uses=1]
	%imm_val = inttoptr i32 %imm_or to %Value		; <%Value> [#uses=1]
	ret %Value %imm_val

NumberLiteralIntSlow:		; preds = %no_exception7
	%toVal = call %Value @X4( i32 %addconv82 )		; <%Value> [#uses=1]
	ret %Value %toVal
}
