; RUN: llvm-as < %s | llc | grep abort | count 1
; Calls to abort should all be merged

; ModuleID = '5898899.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"
	%struct.BoundaryAlignment = type { [3 x i8], i8, i16, i16, i8, [2 x i8] }

define void @passing2(i64 %str.0, i64 %str.1, i16 signext  %s, i32 %j, i8 signext  %c, i16 signext  %t, i16 signext  %u, i8 signext  %d) nounwind  {
entry:
	%str_addr = alloca %struct.BoundaryAlignment		; <%struct.BoundaryAlignment*> [#uses=7]
	%s_addr = alloca i16		; <i16*> [#uses=1]
	%j_addr = alloca i32		; <i32*> [#uses=2]
	%c_addr = alloca i8		; <i8*> [#uses=2]
	%t_addr = alloca i16		; <i16*> [#uses=2]
	%u_addr = alloca i16		; <i16*> [#uses=2]
	%d_addr = alloca i8		; <i8*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = bitcast %struct.BoundaryAlignment* %str_addr to { i64, i64 }*		; <{ i64, i64 }*> [#uses=1]
	%tmp1 = getelementptr { i64, i64 }* %tmp, i32 0, i32 0		; <i64*> [#uses=1]
	store i64 %str.0, i64* %tmp1
	%tmp2 = bitcast %struct.BoundaryAlignment* %str_addr to { i64, i64 }*		; <{ i64, i64 }*> [#uses=1]
	%tmp3 = getelementptr { i64, i64 }* %tmp2, i32 0, i32 1		; <i64*> [#uses=1]
	%bc = bitcast i64* %tmp3 to i8*		; <i8*> [#uses=2]
	%byte = trunc i64 %str.1 to i8		; <i8> [#uses=1]
	store i8 %byte, i8* %bc
	%shft = lshr i64 %str.1, 8		; <i64> [#uses=2]
	%Loc = getelementptr i8* %bc, i32 1		; <i8*> [#uses=2]
	%byte4 = trunc i64 %shft to i8		; <i8> [#uses=1]
	store i8 %byte4, i8* %Loc
	%shft5 = lshr i64 %shft, 8		; <i64> [#uses=2]
	%Loc6 = getelementptr i8* %Loc, i32 1		; <i8*> [#uses=2]
	%byte7 = trunc i64 %shft5 to i8		; <i8> [#uses=1]
	store i8 %byte7, i8* %Loc6
	%shft8 = lshr i64 %shft5, 8		; <i64> [#uses=2]
	%Loc9 = getelementptr i8* %Loc6, i32 1		; <i8*> [#uses=2]
	%byte10 = trunc i64 %shft8 to i8		; <i8> [#uses=1]
	store i8 %byte10, i8* %Loc9
	%shft11 = lshr i64 %shft8, 8		; <i64> [#uses=0]
	%Loc12 = getelementptr i8* %Loc9, i32 1		; <i8*> [#uses=0]
	store i16 %s, i16* %s_addr
	store i32 %j, i32* %j_addr
	store i8 %c, i8* %c_addr
	store i16 %t, i16* %t_addr
	store i16 %u, i16* %u_addr
	store i8 %d, i8* %d_addr
	%tmp13 = getelementptr %struct.BoundaryAlignment* %str_addr, i32 0, i32 0		; <[3 x i8]*> [#uses=1]
	%tmp1314 = bitcast [3 x i8]* %tmp13 to i32*		; <i32*> [#uses=1]
	%tmp15 = load i32* %tmp1314, align 4		; <i32> [#uses=1]
	%tmp16 = shl i32 %tmp15, 14		; <i32> [#uses=1]
	%tmp17 = ashr i32 %tmp16, 23		; <i32> [#uses=1]
	%tmp1718 = trunc i32 %tmp17 to i16		; <i16> [#uses=1]
	%sextl = shl i16 %tmp1718, 7		; <i16> [#uses=1]
	%sextr = ashr i16 %sextl, 7		; <i16> [#uses=2]
	%sextl19 = shl i16 %sextr, 7		; <i16> [#uses=1]
	%sextr20 = ashr i16 %sextl19, 7		; <i16> [#uses=0]
	%sextl21 = shl i16 %sextr, 7		; <i16> [#uses=1]
	%sextr22 = ashr i16 %sextl21, 7		; <i16> [#uses=1]
	%sextr2223 = sext i16 %sextr22 to i32		; <i32> [#uses=1]
	%tmp24 = load i32* %j_addr, align 4		; <i32> [#uses=1]
	%tmp25 = icmp ne i32 %sextr2223, %tmp24		; <i1> [#uses=1]
	%tmp2526 = zext i1 %tmp25 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp2526, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb, label %bb27

bb:		; preds = %entry
	call void (...)* @abort( ) noreturn nounwind 
	unreachable

bb27:		; preds = %entry
	%tmp28 = getelementptr %struct.BoundaryAlignment* %str_addr, i32 0, i32 1		; <i8*> [#uses=1]
	%tmp29 = load i8* %tmp28, align 4		; <i8> [#uses=1]
	%tmp30 = load i8* %c_addr, align 1		; <i8> [#uses=1]
	%tmp31 = icmp ne i8 %tmp29, %tmp30		; <i1> [#uses=1]
	%tmp3132 = zext i1 %tmp31 to i8		; <i8> [#uses=1]
	%toBool33 = icmp ne i8 %tmp3132, 0		; <i1> [#uses=1]
	br i1 %toBool33, label %bb34, label %bb35

bb34:		; preds = %bb27
	call void (...)* @abort( ) noreturn nounwind 
	unreachable

bb35:		; preds = %bb27
	%tmp36 = getelementptr %struct.BoundaryAlignment* %str_addr, i32 0, i32 2		; <i16*> [#uses=1]
	%tmp37 = load i16* %tmp36, align 4		; <i16> [#uses=1]
	%tmp38 = shl i16 %tmp37, 7		; <i16> [#uses=1]
	%tmp39 = ashr i16 %tmp38, 7		; <i16> [#uses=1]
	%sextl40 = shl i16 %tmp39, 7		; <i16> [#uses=1]
	%sextr41 = ashr i16 %sextl40, 7		; <i16> [#uses=2]
	%sextl42 = shl i16 %sextr41, 7		; <i16> [#uses=1]
	%sextr43 = ashr i16 %sextl42, 7		; <i16> [#uses=0]
	%sextl44 = shl i16 %sextr41, 7		; <i16> [#uses=1]
	%sextr45 = ashr i16 %sextl44, 7		; <i16> [#uses=1]
	%tmp46 = load i16* %t_addr, align 2		; <i16> [#uses=1]
	%tmp47 = icmp ne i16 %sextr45, %tmp46		; <i1> [#uses=1]
	%tmp4748 = zext i1 %tmp47 to i8		; <i8> [#uses=1]
	%toBool49 = icmp ne i8 %tmp4748, 0		; <i1> [#uses=1]
	br i1 %toBool49, label %bb50, label %bb51

bb50:		; preds = %bb35
	call void (...)* @abort( ) noreturn nounwind 
	unreachable

bb51:		; preds = %bb35
	%tmp52 = getelementptr %struct.BoundaryAlignment* %str_addr, i32 0, i32 3		; <i16*> [#uses=1]
	%tmp53 = load i16* %tmp52, align 4		; <i16> [#uses=1]
	%tmp54 = shl i16 %tmp53, 7		; <i16> [#uses=1]
	%tmp55 = ashr i16 %tmp54, 7		; <i16> [#uses=1]
	%sextl56 = shl i16 %tmp55, 7		; <i16> [#uses=1]
	%sextr57 = ashr i16 %sextl56, 7		; <i16> [#uses=2]
	%sextl58 = shl i16 %sextr57, 7		; <i16> [#uses=1]
	%sextr59 = ashr i16 %sextl58, 7		; <i16> [#uses=0]
	%sextl60 = shl i16 %sextr57, 7		; <i16> [#uses=1]
	%sextr61 = ashr i16 %sextl60, 7		; <i16> [#uses=1]
	%tmp62 = load i16* %u_addr, align 2		; <i16> [#uses=1]
	%tmp63 = icmp ne i16 %sextr61, %tmp62		; <i1> [#uses=1]
	%tmp6364 = zext i1 %tmp63 to i8		; <i8> [#uses=1]
	%toBool65 = icmp ne i8 %tmp6364, 0		; <i1> [#uses=1]
	br i1 %toBool65, label %bb66, label %bb67

bb66:		; preds = %bb51
	call void (...)* @abort( ) noreturn nounwind 
	unreachable

bb67:		; preds = %bb51
	%tmp68 = getelementptr %struct.BoundaryAlignment* %str_addr, i32 0, i32 4		; <i8*> [#uses=1]
	%tmp69 = load i8* %tmp68, align 4		; <i8> [#uses=1]
	%tmp70 = load i8* %d_addr, align 1		; <i8> [#uses=1]
	%tmp71 = icmp ne i8 %tmp69, %tmp70		; <i1> [#uses=1]
	%tmp7172 = zext i1 %tmp71 to i8		; <i8> [#uses=1]
	%toBool73 = icmp ne i8 %tmp7172, 0		; <i1> [#uses=1]
	br i1 %toBool73, label %bb74, label %bb75

bb74:		; preds = %bb67
	call void (...)* @abort( ) noreturn nounwind 
	unreachable

bb75:		; preds = %bb67
	br label %return

return:		; preds = %bb75
	ret void
}

declare void @abort(...) noreturn nounwind 
