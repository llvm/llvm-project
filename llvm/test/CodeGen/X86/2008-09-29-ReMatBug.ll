; RUN: llc < %s -mtriple=i386-apple-darwin -relocation-model=pic -frame-pointer=all

	%struct..0objc_selector = type opaque
	%struct.NSString = type opaque
	%struct.XCStringList = type { i32, %struct._XCStringListNode* }
	%struct._XCStringListNode = type { [3 x i8], [0 x i8], i8 }
	%struct.__builtin_CFString = type { i32*, i32, i8*, i32 }
@0 = internal constant %struct.__builtin_CFString { i32* getelementptr ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr ([3 x i8], [3 x i8]* @"\01LC", i32 0, i32 0), i32 2 }		; <%struct.__builtin_CFString*>:0 [#uses=1]
@__CFConstantStringClassReference = external global [0 x i32]		; <[0 x i32]*> [#uses=1]
@"\01LC" = internal constant [3 x i8] c"NO\00"		; <[3 x i8]*> [#uses=1]
@"\01LC1" = internal constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=1]
@llvm.used1 = appending global [1 x i8*] [ i8* bitcast (%struct.NSString* (%struct.XCStringList*, %struct..0objc_selector*)* @"-[XCStringList stringRepresentation]" to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define %struct.NSString* @"-[XCStringList stringRepresentation]"(%struct.XCStringList* %self, %struct..0objc_selector* %_cmd) nounwind {
entry:
	%0 = load i32, i32* null, align 4		; <i32> [#uses=1]
	%1 = and i32 %0, 16777215		; <i32> [#uses=1]
	%2 = icmp eq i32 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %bb44, label %bb4

bb4:		; preds = %entry
	%3 = load %struct._XCStringListNode*, %struct._XCStringListNode** null, align 4		; <%struct._XCStringListNode*> [#uses=2]
	%4 = icmp eq %struct._XCStringListNode* %3, null		; <i1> [#uses=1]
	%5 = bitcast %struct._XCStringListNode* %3 to i32*		; <i32*> [#uses=1]
	br label %bb37.outer

bb6:		; preds = %bb37
	br label %bb19

bb19:		; preds = %bb37, %bb6
	%.rle = phi i32 [ 0, %bb6 ], [ %10, %bb37 ]		; <i32> [#uses=1]
	%bufptr.0.lcssa = phi i8* [ null, %bb6 ], [ null, %bb37 ]		; <i8*> [#uses=2]
	%6 = and i32 %.rle, 16777215		; <i32> [#uses=1]
	%7 = icmp eq i32 %6, 0		; <i1> [#uses=1]
	br i1 %7, label %bb25.split, label %bb37

bb25.split:		; preds = %bb19
	call void @foo(i8* getelementptr ([1 x i8], [1 x i8]* @"\01LC1", i32 0, i32 0)) nounwind nounwind
	br label %bb35.outer

bb34:		; preds = %bb35, %bb35, %bb35, %bb35
	%8 = getelementptr i8, i8* %bufptr.0.lcssa, i32 %totalLength.0.ph		; <i8*> [#uses=1]
	store i8 92, i8* %8, align 1
	br label %bb35.outer

bb35.outer:		; preds = %bb34, %bb25.split
	%totalLength.0.ph = add i32 0, %totalLength.1.ph		; <i32> [#uses=2]
	br label %bb35

bb35:		; preds = %bb35, %bb35.outer
	%9 = load i8, i8* null, align 1		; <i8> [#uses=1]
	switch i8 %9, label %bb35 [
		i8 0, label %bb37.outer
		i8 32, label %bb34
		i8 92, label %bb34
		i8 34, label %bb34
		i8 39, label %bb34
	]

bb37.outer:		; preds = %bb35, %bb4
	%totalLength.1.ph = phi i32 [ 0, %bb4 ], [ %totalLength.0.ph, %bb35 ]		; <i32> [#uses=1]
	%bufptr.1.ph = phi i8* [ null, %bb4 ], [ %bufptr.0.lcssa, %bb35 ]		; <i8*> [#uses=2]
	br i1 %4, label %bb39.split, label %bb37

bb37:		; preds = %bb37.outer, %bb19
	%10 = load i32, i32* %5, align 4		; <i32> [#uses=1]
	br i1 false, label %bb6, label %bb19

bb39.split:		; preds = %bb37.outer
	%11 = bitcast i8* null to %struct.NSString*		; <%struct.NSString*> [#uses=2]
	%12 = icmp eq i8* null, %bufptr.1.ph		; <i1> [#uses=1]
	br i1 %12, label %bb44, label %bb42

bb42:		; preds = %bb39.split
	call void @quux(i8* %bufptr.1.ph) nounwind nounwind
	ret %struct.NSString* %11

bb44:		; preds = %bb39.split, %entry
	%.0 = phi %struct.NSString* [ bitcast (%struct.__builtin_CFString* @0 to %struct.NSString*), %entry ], [ %11, %bb39.split ]		; <%struct.NSString*> [#uses=1]
	ret %struct.NSString* %.0
}

declare void @foo(i8*)

declare void @quux(i8*)
