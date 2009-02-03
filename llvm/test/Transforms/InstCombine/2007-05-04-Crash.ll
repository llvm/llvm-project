; RUN: llvm-as < %s | opt -instcombine -disable-output
; PR1384

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
	%struct.CFRuntimeBase = type { i32, [4 x i8] }
	%struct.CGColor = type opaque
	%struct.CGColorSpace = type { %struct.CFRuntimeBase, i8, i8, i8, i32, i32, i32, %struct.CGColor*, float*, %struct.CGMD5Signature, %struct.CGMD5Signature*, [0 x %struct.CGColorSpaceDescriptor] }
	%struct.CGColorSpaceCalibratedRGBData = type { [3 x float], [3 x float], [3 x float], [9 x float] }
	%struct.CGColorSpaceDescriptor = type { %struct.CGColorSpaceCalibratedRGBData }
	%struct.CGColorSpaceLabData = type { [3 x float], [3 x float], [4 x float] }
	%struct.CGMD5Signature = type { [16 x i8], i8 }

declare fastcc %struct.CGColorSpace* @CGColorSpaceCreate(i32, i32)

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

define %struct.CGColorSpace* @CGColorSpaceCreateLab(float* %whitePoint, float* %blackPoint, float* %range) {
entry:
	%tmp17 = call fastcc %struct.CGColorSpace* @CGColorSpaceCreate( i32 5, i32 3 )		; <%struct.CGColorSpace*> [#uses=2]
	%tmp28 = getelementptr %struct.CGColorSpace* %tmp17, i32 0, i32 11		; <[0 x %struct.CGColorSpaceDescriptor]*> [#uses=1]
	%tmp29 = getelementptr [0 x %struct.CGColorSpaceDescriptor]* %tmp28, i32 0, i32 0		; <%struct.CGColorSpaceDescriptor*> [#uses=1]
	%tmp30 = getelementptr %struct.CGColorSpaceDescriptor* %tmp29, i32 0, i32 0		; <%struct.CGColorSpaceCalibratedRGBData*> [#uses=1]
	%tmp3031 = bitcast %struct.CGColorSpaceCalibratedRGBData* %tmp30 to %struct.CGColorSpaceLabData*		; <%struct.CGColorSpaceLabData*> [#uses=1]
	%tmp45 = getelementptr %struct.CGColorSpaceLabData* %tmp3031, i32 0, i32 2		; <[4 x float]*> [#uses=1]
	%tmp46 = getelementptr [4 x float]* %tmp45, i32 0, i32 0		; <float*> [#uses=1]
	%tmp4648 = bitcast float* %tmp46 to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %tmp4648, i8* null, i32 16, i32 4 )
	ret %struct.CGColorSpace* %tmp17
}
