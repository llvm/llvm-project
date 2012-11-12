; RUN: llc < %s -mtriple=arm-none-linux-gnueabi
;pr3502

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
	%struct.ArmPTD = type { i32 }
	%struct.RegisterSave = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.SHARED_AREA = type { i32, %struct.SHARED_AREA*, %struct.SHARED_AREA*, %struct.SHARED_AREA*, %struct.ArmPTD, void (%struct.RegisterSave*)*, void (%struct.RegisterSave*)*, i32, [1024 x i8], i32, i32, i32, i32, i32, i8, i8, i16, i32, i32, i32, i32, [16 x i8], i32, i32, i32, i8, i8, i8, i32, i16, i32, i64, i32, i32, i32, i32, i32, i32, i8*, i32, [256 x i8], i32, i32, i32, [20 x i8], %struct.RegisterSave, { %struct.WorldSwitchV5 }, [4 x i32] }
	%struct.WorldSwitchV5 = type { i32, i32, i32, i32, i32, i32, i32 }

define void @SomeCall(i32 %num) nounwind {
entry:
	tail call void asm sideeffect "mcr p15, 0, $0, c7, c10, 4 \0A\09", "r,~{memory}"(i32 0) nounwind
	tail call void asm sideeffect "mcr p15,0,$0,c7,c14,0", "r,~{memory}"(i32 0) nounwind
	%0 = load %struct.SHARED_AREA** null, align 4		; <%struct.SHARED_AREA*> [#uses=1]
	%1 = ptrtoint %struct.SHARED_AREA* %0 to i32		; <i32> [#uses=1]
	%2 = lshr i32 %1, 20		; <i32> [#uses=1]
	%3 = tail call i32 @SetCurrEntry(i32 %2, i32 0) nounwind		; <i32> [#uses=0]
	tail call void @ClearStuff(i32 0) nounwind
	ret void
}

declare i32 @SetCurrEntry(i32, i32)

declare void @ClearStuff(i32)
