; RUN: llc < %s
; PR3450

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
	%struct.BitMap = type { i8* }
	%struct.BitMapListStruct = type { %struct.BitMap, %struct.BitMapListStruct*, %struct.BitMapListStruct* }
	%struct.Material = type { float, float, float, %struct.Material*, %struct.Material* }
	%struct.ObjPoint = type { double, double, double, double, double, double }
	%struct.ObjectStruct = type { [57 x i8], %struct.PointListStruct*, %struct.Poly3Struct*, %struct.Poly4Struct*, %struct.Texture*, %struct.Material*, %struct.Point, i32, i32, %struct.Point, %struct.Point, %struct.Point, %struct.ObjectStruct*, %struct.ObjectStruct*, i32, i32, i32, i32, i32, i32, i32, %struct.ObjectStruct*, %struct.ObjectStruct* }
	%struct.Point = type { double, double, double }
	%struct.PointListStruct = type { %struct.ObjPoint*, %struct.PointListStruct*, %struct.PointListStruct* }
	%struct.Poly3Struct = type { [3 x %struct.ObjPoint*], %struct.Material*, %struct.Texture*, %struct.Poly3Struct*, %struct.Poly3Struct* }
	%struct.Poly4Struct = type { [4 x %struct.ObjPoint*], %struct.Material*, %struct.Texture*, %struct.Poly4Struct*, %struct.Poly4Struct* }
	%struct.Texture = type { %struct.Point, %struct.BitMapListStruct*, %struct.Point, %struct.Point, %struct.Point, %struct.Texture*, %struct.Texture* }

define fastcc void @ScaleObjectAdd(%struct.ObjectStruct* %o, double %sx, double %sy, double %sz) nounwind {
entry:
	%sz101112.ins = or i960 0, 0		; <i960> [#uses=1]
	br i1 false, label %return, label %bb1.preheader

bb1.preheader:		; preds = %entry
	%0 = lshr i960 %sz101112.ins, 640		; <i960> [#uses=0]
	br label %bb1

bb1:		; preds = %bb1, %bb1.preheader
	br label %bb1

return:		; preds = %entry
	ret void
}
