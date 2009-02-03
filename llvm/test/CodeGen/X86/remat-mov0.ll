; RUN: llvm-as < %s | llc -march=x86 | grep xor | count 2

	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.ImgT = type { i8, i8*, i8*, %struct.FILE*, i32, i32, i32, i32, i8*, double*, float*, float*, float*, i32*, double, double, i32*, double*, i32*, i32* }
	%struct._CompT = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, i8, %struct._PixT*, %struct._CompT*, i8, %struct._CompT* }
	%struct._PixT = type { i32, i32, %struct._PixT* }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }

declare fastcc void @MergeComponents(%struct._CompT*, %struct._CompT*, %struct._CompT*, %struct._CompT**, %struct.ImgT*) nounwind 

define fastcc void @MergeToLeft(%struct._CompT* %comp, %struct._CompT** %head, %struct.ImgT* %img) nounwind  {
entry:
	br label %bb208

bb105:		; preds = %bb200
	br i1 false, label %bb197, label %bb149

bb149:		; preds = %bb105
	%tmp151 = getelementptr %struct._CompT* %comp, i32 0, i32 0		; <i32*> [#uses=1]
	br label %bb193

bb193:		; preds = %bb184, %bb149
	%tmp196 = load i32* %tmp151, align 4		; <i32> [#uses=1]
	br label %bb197

bb197:		; preds = %bb193, %bb105
	%last_comp.0 = phi i32 [ %tmp196, %bb193 ], [ 0, %bb105 ]		; <i32> [#uses=0]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb200

bb200:		; preds = %bb208, %bb197
	%indvar = phi i32 [ 0, %bb208 ], [ %indvar.next, %bb197 ]		; <i32> [#uses=2]
	%xm.0 = sub i32 %indvar, 0		; <i32> [#uses=1]
	%tmp202 = icmp slt i32 %xm.0, 1		; <i1> [#uses=1]
	br i1 %tmp202, label %bb105, label %bb208

bb208:		; preds = %bb200, %entry
	br label %bb200
}
