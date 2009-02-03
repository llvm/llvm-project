; RUN: llvm-as < %s | opt -loop-rotate -disable-output
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9"
	%struct.NSArray = type { %struct.NSObject }
	%struct.NSObject = type { %struct.objc_class* }
	%struct.NSRange = type { i64, i64 }
	%struct._message_ref_t = type { %struct.NSObject* (%struct.NSObject*, %struct._message_ref_t*, ...)*, %struct.objc_selector* }
	%struct.objc_class = type opaque
	%struct.objc_selector = type opaque
@"\01L_OBJC_MESSAGE_REF_26" = external global %struct._message_ref_t		; <%struct._message_ref_t*> [#uses=1]

define %struct.NSArray* @"-[NSString(DocSetPrivateAddition) _dsa_stringAsPathComponent]"(%struct.NSArray* %self, %struct._message_ref_t* %_cmd) {
entry:
	br label %bb116

bb116:		; preds = %bb131, %entry
	%tmp123 = call %struct.NSRange null( %struct.NSObject* null, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_26", %struct.NSArray* null )		; <%struct.NSRange> [#uses=1]
	br i1 false, label %bb141, label %bb131

bb131:		; preds = %bb116
	%mrv_gr125 = getresult %struct.NSRange %tmp123, 1		; <i64> [#uses=0]
	br label %bb116

bb141:		; preds = %bb116
	ret %struct.NSArray* null
}
