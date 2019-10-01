; RUN: llc < %s -mtriple=x86_64-apple-darwin
; RUN: llc < %s -mtriple=x86_64-apple-darwin -relocation-model=pic -frame-pointer=all -O0 -regalloc=fast
; PR5534

	%struct.CGPoint = type { double, double }
	%struct.NSArray = type { %struct.NSObject }
	%struct.NSAssertionHandler = type { %struct.NSObject, i8* }
	%struct.NSDockTile = type { %struct.NSObject, %struct.NSObject*, i8*, %struct.NSView*, %struct.NSView*, %struct.NSView*, %struct.NSArray*, %struct._SPFlags, %struct.CGPoint, [5 x %struct.NSObject*] }
	%struct.NSDocument = type { %struct.NSObject, %struct.NSWindow*, %struct.NSObject*, %struct.NSURL*, %struct.NSArray*, %struct.NSPrintInfo*, i64, %struct.NSView*, %struct.NSObject*, %struct.NSObject*, %struct.NSUndoManager*, %struct._BCFlags2, %struct.NSArray* }
	%struct.AA = type { %struct.NSObject, %struct.NSDocument*, %struct.NSURL*, %struct.NSArray*, %struct.NSArray* }
	%struct.NSError = type { %struct.NSObject, i8*, i64, %struct.NSArray*, %struct.NSArray* }
	%struct.NSImage = type { %struct.NSObject, %struct.NSArray*, %struct.CGPoint, %struct._BCFlags2, %struct.NSObject*, %struct._NSImageAuxiliary* }
	%struct.NSMutableArray = type { %struct.NSArray }
	%struct.NSObject = type { %struct.NSObject* }
	%struct.NSPrintInfo = type { %struct.NSObject, %struct.NSMutableArray*, %struct.NSObject* }
	%struct.NSRect = type { %struct.CGPoint, %struct.CGPoint }
	%struct.NSRegion = type opaque
	%struct.NSResponder = type { %struct.NSObject, %struct.NSObject* }
	%struct.NSToolbar = type { %struct.NSObject, %struct.NSArray*, %struct.NSMutableArray*, %struct.NSMutableArray*, %struct.NSArray*, %struct.NSObject*, %struct.NSArray*, i8*, %struct.NSObject*, %struct.NSWindow*, %struct.NSObject*, %struct.NSObject*, i64, %struct._BCFlags2, i64, %struct.NSObject* }
	%struct.NSURL = type { %struct.NSObject, %struct.NSArray*, %struct.NSURL*, i8*, i8* }
	%struct.NSUndoManager = type { %struct.NSObject, %struct.NSObject*, %struct.NSObject*, %struct.NSArray*, i64, %struct._SPFlags, %struct.NSObject*, i8*, i8*, i8* }
	%struct.NSView = type { %struct.NSResponder, %struct.NSRect, %struct.NSRect, %struct.NSObject*, %struct.NSObject*, %struct.NSWindow*, %struct.NSObject*, %struct.NSObject*, %struct.NSObject*, %struct.NSObject*, %struct._NSViewAuxiliary*, %struct._BCFlags, %struct._SPFlags }
	%struct.NSWindow = type { %struct.NSResponder, %struct.NSRect, %struct.NSObject*, %struct.NSObject*, %struct.NSResponder*, %struct.NSView*, %struct.NSView*, %struct.NSObject*, %struct.NSObject*, i32, i64, i32, %struct.NSArray*, %struct.NSObject*, i8, i8, i8, i8, i8*, i8*, %struct.NSImage*, i32, %struct.NSMutableArray*, %struct.NSURL*, %struct.CGPoint*, %struct.NSArray*, %struct.NSArray*, %struct.__wFlags, %struct.NSObject*, %struct.NSView*, %struct.NSWindowAuxiliary* }
	%struct.NSWindowAuxiliary = type { %struct.NSObject, %struct.NSArray*, %struct.NSDockTile*, %struct._NSWindowAnimator*, %struct.NSRect, i32, %struct.NSAssertionHandler*, %struct.NSUndoManager*, %struct.NSWindowController*, %struct.NSAssertionHandler*, %struct.NSObject*, i32, %struct.__CFRunLoopObserver*, %struct.__CFRunLoopObserver*, %struct.NSArray*, %struct.NSArray*, %struct.NSView*, %struct.NSRegion*, %struct.NSWindow*, %struct.NSWindow*, %struct.NSArray*, %struct.NSMutableArray*, %struct.NSArray*, %struct.NSWindow*, %struct.CGPoint, %struct.NSObject*, i8*, i8*, i32, %struct.NSObject*, %struct.NSArray*, double, %struct.CGPoint, %struct.NSArray*, %struct.NSMutableArray*, %struct.NSMutableArray*, %struct.NSWindow*, %struct.NSView*, %struct.NSArray*, %struct.__auxWFlags, i32, i8*, double, %struct.NSObject*, %struct.NSObject*, %struct.__CFArray*, %struct.NSRegion*, %struct.NSArray*, %struct.NSRect, %struct.NSToolbar*, %struct.NSRect, %struct.NSMutableArray* }
	%struct.NSWindowController = type { %struct.NSResponder, %struct.NSWindow*, %struct.NSArray*, %struct.NSDocument*, %struct.NSArray*, %struct.NSObject*, %struct._SPFlags, %struct.NSArray*, %struct.NSObject* }
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
	%struct._message_ref_t = type { %struct.NSObject* (%struct.NSObject*, %struct._message_ref_t*, ...)*, %struct.objc_selector* }
	%struct.objc_selector = type opaque
@"\01L_OBJC_MESSAGE_REF_228" = internal global %struct._message_ref_t zeroinitializer		; <%struct._message_ref_t*> [#uses=1]
@llvm.used1 = appending global [1 x i8*] [ i8* bitcast (void (%struct.AA*, %struct._message_ref_t*, %struct.NSError*, i64, %struct.NSObject*, %struct.objc_selector*, i8*)* @"-[AA BB:optionIndex:delegate:CC:contextInfo:]" to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define void @"-[AA BB:optionIndex:delegate:CC:contextInfo:]"(%struct.AA* %self, %struct._message_ref_t* %_cmd, %struct.NSError* %inError, i64 %inOptionIndex, %struct.NSObject* %inDelegate, %struct.objc_selector* %inDidRecoverSelector, i8* %inContextInfo) {
entry:
	%tmp105 = load %struct.NSArray*, %struct.NSArray** null, align 8		; <%struct.NSArray*> [#uses=1]
	%tmp107 = load %struct.NSObject*, %struct.NSObject** null, align 8		; <%struct.NSObject*> [#uses=1]
	call void null( %struct.NSObject* %tmp107, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_228", %struct.NSArray* %tmp105, i8 signext  0 )
	%tmp111 = call %struct.NSObject* (%struct.NSObject*, %struct.objc_selector*, ...) @objc_msgSend( %struct.NSObject* null, %struct.objc_selector* null, i32 0, i8* null )		; <%struct.NSObject*> [#uses=0]
	ret void
}

declare %struct.NSObject* @objc_msgSend(%struct.NSObject*, %struct.objc_selector*, ...)
