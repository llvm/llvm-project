; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -regalloc=fast -optimize-regalloc=0 -relocation-model=pic | FileCheck %s

	%struct.NSError = type opaque
	%struct.NSManagedObjectContext = type opaque
	%struct.NSString = type opaque
	%struct.NSURL = type opaque
	%struct._message_ref_t = type { %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)*, %struct.objc_selector* }
	%struct.objc_object = type {  }
	%struct.objc_selector = type opaque
@"\01L_OBJC_MESSAGE_REF_2" = external global %struct._message_ref_t		; <%struct._message_ref_t*> [#uses=2]
@"\01L_OBJC_MESSAGE_REF_6" = external global %struct._message_ref_t		; <%struct._message_ref_t*> [#uses=2]
@NSXMLStoreType = external constant %struct.NSString*		; <%struct.NSString**> [#uses=1]
@"\01L_OBJC_MESSAGE_REF_4" = external global %struct._message_ref_t		; <%struct._message_ref_t*> [#uses=2]

; TODO: KB: ORiginal test case was just checking it compiles; is this worth keeping?
; CHECK: managedObjectContextWithModelURL
; CHECK-NOT: blr
; CHECK: .cfi_endproc

define %struct.NSManagedObjectContext* @"+[ListGenerator(Private) managedObjectContextWithModelURL:storeURL:]"(%struct.objc_object* %self, %struct._message_ref_t* %_cmd, %struct.NSURL* %modelURL, %struct.NSURL* %storeURL) {
entry:
	%tmp27 = load %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)*, %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)** getelementptr (%struct._message_ref_t, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_2", i32 0, i32 0), align 8		; <%struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)*> [#uses=1]
	%tmp29 = call %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...) %tmp27( %struct.objc_object* null, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_2" )		; <%struct.objc_object*> [#uses=0]
	%tmp33 = load %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)*, %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)** getelementptr (%struct._message_ref_t, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_6", i32 0, i32 0), align 8		; <%struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)*> [#uses=1]
	%tmp34 = load %struct.NSString*, %struct.NSString** @NSXMLStoreType, align 8		; <%struct.NSString*> [#uses=1]
	%tmp40 = load %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)*, %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)** getelementptr (%struct._message_ref_t, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_4", i32 0, i32 0), align 8		; <%struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)*> [#uses=1]
	%tmp42 = call %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...) %tmp40( %struct.objc_object* null, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_4", i32 1 )		; <%struct.objc_object*> [#uses=0]
	%tmp48 = call %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...) %tmp33( %struct.objc_object* null, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_6", %struct.NSString* %tmp34, i8* null, %struct.NSURL* null, %struct.objc_object* null, %struct.NSError** null )		; <%struct.objc_object*> [#uses=0]
	unreachable
}
