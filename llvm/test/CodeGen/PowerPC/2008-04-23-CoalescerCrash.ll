; RUN: llc < %s -mtriple=powerpc-apple-darwin

@_ZL10DeviceCode = internal global i16 0		; <i16*> [#uses=1]
@.str19 = internal constant [64 x i8] c"unlock_then_erase_sector: failed to erase block (status= 0x%x)\0A\00"		; <[64 x i8]*> [#uses=1]
@.str34 = internal constant [68 x i8] c"ProgramByWords - Erasing sector 0x%llx to 0x%llx (size 0x%x bytes)\0A\00"		; <[68 x i8]*> [#uses=1]
@.str35 = internal constant [37 x i8] c"ProgramByWords - Done erasing flash\0A\00"		; <[37 x i8]*> [#uses=1]
@.str36 = internal constant [48 x i8] c"ProgramByWords - Starting to write to FLASH...\0A\00"		; <[48 x i8]*> [#uses=1]

declare void @IOLog(i8*, ...)

declare void @IODelay(i32)

define i32 @_Z14ProgramByWordsPvyy(i8* %buffer, i64 %Offset, i64 %bufferSize) nounwind  {
entry:
	store volatile i8 -1, i8* null, align 1
	%tmp28 = icmp eq i8 0, 0		; <i1> [#uses=1]
	br i1 %tmp28, label %bb107, label %bb

bb:		; preds = %entry
	%tmp9596430 = zext i32 0 to i64		; <i64> [#uses=1]
	%tmp98431 = add i64 %tmp9596430, %Offset		; <i64> [#uses=1]
	%tmp100433 = icmp ugt i64 %tmp98431, %Offset		; <i1> [#uses=1]
	br i1 %tmp100433, label %bb31, label %bb103

bb31:		; preds = %_Z24unlock_then_erase_sectory.exit, %bb
	%Pos.0.reg2mem.0 = phi i64 [ %tmp93, %_Z24unlock_then_erase_sectory.exit ], [ %Offset, %bb ]		; <i64> [#uses=3]
	%tmp35 = load i16* @_ZL10DeviceCode, align 2		; <i16> [#uses=1]
	%tmp3536 = zext i16 %tmp35 to i32		; <i32> [#uses=2]
	%tmp37 = and i32 %tmp3536, 65520		; <i32> [#uses=1]
	%tmp38 = icmp eq i32 %tmp37, 35008		; <i1> [#uses=1]
	%tmp34 = sub i64 %Pos.0.reg2mem.0, %Offset		; <i64> [#uses=2]
	br i1 %tmp38, label %bb41, label %bb68

bb41:		; preds = %bb31
	%tmp43 = add i32 0, -1		; <i32> [#uses=1]
	%tmp4344 = zext i32 %tmp43 to i64		; <i64> [#uses=1]
	%tmp46 = and i64 %tmp4344, %tmp34		; <i64> [#uses=0]
	%tmp49 = and i32 %tmp3536, 1		; <i32> [#uses=0]
	ret i32 0

bb68:		; preds = %bb31
	tail call void (i8*, ...)* @IOLog( i8* getelementptr ([68 x i8]* @.str34, i32 0, i32 0), i64 %tmp34, i64 0, i32 131072 ) nounwind 
	%tmp2021.i = trunc i64 %Pos.0.reg2mem.0 to i32		; <i32> [#uses=1]
	%tmp202122.i = inttoptr i32 %tmp2021.i to i8*		; <i8*> [#uses=1]
	tail call void @IODelay( i32 500 ) nounwind 
	%tmp53.i = load volatile i16* null, align 2		; <i16> [#uses=2]
	%tmp5455.i = zext i16 %tmp53.i to i32		; <i32> [#uses=1]
	br i1 false, label %bb.i, label %bb65.i

bb.i:		; preds = %bb68
	ret i32 0

bb65.i:		; preds = %bb68
	%tmp67.i = icmp eq i16 %tmp53.i, 128		; <i1> [#uses=1]
	br i1 %tmp67.i, label %_Z24unlock_then_erase_sectory.exit, label %bb70.i

bb70.i:		; preds = %bb65.i
	tail call void (i8*, ...)* @IOLog( i8* getelementptr ([64 x i8]* @.str19, i32 0, i32 0), i32 %tmp5455.i ) nounwind 
	ret i32 0

_Z24unlock_then_erase_sectory.exit:		; preds = %bb65.i
	store volatile i8 -1, i8* %tmp202122.i, align 1
	%tmp93 = add i64 0, %Pos.0.reg2mem.0		; <i64> [#uses=2]
	%tmp98 = add i64 0, %Offset		; <i64> [#uses=1]
	%tmp100 = icmp ugt i64 %tmp98, %tmp93		; <i1> [#uses=1]
	br i1 %tmp100, label %bb31, label %bb103

bb103:		; preds = %_Z24unlock_then_erase_sectory.exit, %bb
	tail call void (i8*, ...)* @IOLog( i8* getelementptr ([37 x i8]* @.str35, i32 0, i32 0) ) nounwind 
	ret i32 0

bb107:		; preds = %entry
	tail call void (i8*, ...)* @IOLog( i8* getelementptr ([48 x i8]* @.str36, i32 0, i32 0) ) nounwind 
	%tmp114115 = bitcast i8* %buffer to i16*		; <i16*> [#uses=1]
	%tmp256 = lshr i64 %bufferSize, 1		; <i64> [#uses=1]
	%tmp256257 = trunc i64 %tmp256 to i32		; <i32> [#uses=1]
	%tmp258 = getelementptr i16* %tmp114115, i32 %tmp256257		; <i16*> [#uses=0]
	ret i32 0
}

define i32 @_Z17program_64B_blockyPm(i64 %Base, i32* %pData) nounwind  {
entry:
	unreachable
}

define i32 @_Z15ProgramByBlocksyy(i64 %Offset, i64 %bufferSize) nounwind  {
entry:
	ret i32 0
}
