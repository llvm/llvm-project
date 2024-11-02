; RUN: llc < %s -mtriple=x86_64-apple-darwin
; RUN: llc < %s -mtriple=x86_64-apple-darwin -relocation-model=pic -frame-pointer=all -O0 -regalloc=fast
; PR5534

	%struct.CGPoint = type { double, double }
	%struct.NSArray = type { %struct.NSObject }
	%struct.NSAssertionHandler = type { %struct.NSObject, ptr }
	%struct.NSDockTile = type { %struct.NSObject, ptr, ptr, ptr, ptr, ptr, ptr, %struct._SPFlags, %struct.CGPoint, [5 x ptr] }
	%struct.NSDocument = type { %struct.NSObject, ptr, ptr, ptr, ptr, ptr, i64, ptr, ptr, ptr, ptr, %struct._BCFlags2, ptr }
	%struct.AA = type { %struct.NSObject, ptr, ptr, ptr, ptr }
	%struct.NSError = type { %struct.NSObject, ptr, i64, ptr, ptr }
	%struct.NSImage = type { %struct.NSObject, ptr, %struct.CGPoint, %struct._BCFlags2, ptr, ptr }
	%struct.NSMutableArray = type { %struct.NSArray }
	%struct.NSObject = type { ptr }
	%struct.NSPrintInfo = type { %struct.NSObject, ptr, ptr }
	%struct.NSRect = type { %struct.CGPoint, %struct.CGPoint }
	%struct.NSRegion = type opaque
	%struct.NSResponder = type { %struct.NSObject, ptr }
	%struct.NSToolbar = type { %struct.NSObject, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, %struct._BCFlags2, i64, ptr }
	%struct.NSURL = type { %struct.NSObject, ptr, ptr, ptr, ptr }
	%struct.NSUndoManager = type { %struct.NSObject, ptr, ptr, ptr, i64, %struct._SPFlags, ptr, ptr, ptr, ptr }
	%struct.NSView = type { %struct.NSResponder, %struct.NSRect, %struct.NSRect, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct._BCFlags, %struct._SPFlags }
	%struct.NSWindow = type { %struct.NSResponder, %struct.NSRect, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i64, i32, ptr, ptr, i8, i8, i8, i8, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, %struct.__wFlags, ptr, ptr, ptr }
	%struct.NSWindowAuxiliary = type { %struct.NSObject, ptr, ptr, ptr, %struct.NSRect, i32, ptr, ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.CGPoint, ptr, ptr, ptr, i32, ptr, ptr, double, %struct.CGPoint, ptr, ptr, ptr, ptr, ptr, ptr, %struct.__auxWFlags, i32, ptr, double, ptr, ptr, ptr, ptr, ptr, %struct.NSRect, ptr, %struct.NSRect, ptr }
	%struct.NSWindowController = type { %struct.NSResponder, ptr, ptr, ptr, ptr, ptr, %struct._SPFlags, ptr, ptr }
	%struct._BCFlags = type <{ i8, i8, i8, i8 }>
	%struct._BCFlags2 = type <{ i8, [3 x i8] }>
	%struct._NSImageAuxiliary = type opaque
	%struct._NSViewAuxiliary = type opaque
	%struct._NSWindowAnimator = type opaque
	%struct._SPFlags = type <{ i32 }>
	%struct.__CFArray = type opaque
	%struct.__CFRunLoopObserver = type opaque
	%struct.__auxWFlags = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i16 }
	%struct.__wFlags = type <{ i8, i8, i8, i8, i8, i8, i8, i8 }>
	%struct._message_ref_t = type { ptr, ptr }
	%struct.objc_selector = type opaque
@"\01L_OBJC_MESSAGE_REF_228" = internal global %struct._message_ref_t zeroinitializer		; <ptr> [#uses=1]
@llvm.used1 = appending global [1 x ptr] [ ptr @"-[AA BB:optionIndex:delegate:CC:contextInfo:]" ], section "llvm.metadata"		; <ptr> [#uses=0]

define void @"-[AA BB:optionIndex:delegate:CC:contextInfo:]"(ptr %self, ptr %_cmd, ptr %inError, i64 %inOptionIndex, ptr %inDelegate, ptr %inDidRecoverSelector, ptr %inContextInfo) {
entry:
	%tmp105 = load ptr, ptr null, align 8		; <ptr> [#uses=1]
	%tmp107 = load ptr, ptr null, align 8		; <ptr> [#uses=1]
	call void null( ptr %tmp107, ptr @"\01L_OBJC_MESSAGE_REF_228", ptr %tmp105, i8 signext  0 )
	%tmp111 = call ptr (ptr, ptr, ...) @objc_msgSend( ptr null, ptr null, i32 0, ptr null )		; <ptr> [#uses=0]
	ret void
}

declare ptr @objc_msgSend(ptr, ptr, ...)
