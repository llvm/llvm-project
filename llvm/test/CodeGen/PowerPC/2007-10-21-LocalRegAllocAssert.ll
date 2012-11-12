; RUN: llc < %s -mtriple=powerpc64-apple-darwin9 -regalloc=fast -optimize-regalloc=0 -relocation-model=pic

	%struct.NSError = type opaque
	%struct.NSManagedObjectContext = type opaque
	%struct.NSPersistentStoreCoordinator = type opaque
	%struct.NSString = type opaque
	%struct.NSURL = type opaque
	%struct._message_ref_t = type { %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)*, %struct.objc_selector* }
	%struct.objc_object = type {  }
	%struct.objc_selector = type opaque
@"\01L_OBJC_MESSAGE_REF_2" = external global %struct._message_ref_t		; <%struct._message_ref_t*> [#uses=1]
@"\01L_OBJC_MESSAGE_REF_6" = external global %struct._message_ref_t		; <%struct._message_ref_t*> [#uses=1]
@NSXMLStoreType = external constant %struct.NSString*		; <%struct.NSString**> [#uses=1]
@"\01L_OBJC_MESSAGE_REF_5" = external global %struct._message_ref_t		; <%struct._message_ref_t*> [#uses=2]
@"\01L_OBJC_MESSAGE_REF_4" = external global %struct._message_ref_t		; <%struct._message_ref_t*> [#uses=1]

define %struct.NSManagedObjectContext* @"+[ListGenerator(Private) managedObjectContextWithModelURL:storeURL:]"(%struct.objc_object* %self, %struct._message_ref_t* %_cmd, %struct.NSURL* %modelURL, %struct.NSURL* %storeURL) {
entry:
	%storeCoordinator = alloca %struct.NSPersistentStoreCoordinator*		; <%struct.NSPersistentStoreCoordinator**> [#uses=0]
	%tmp29 = call %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)* null( %struct.objc_object* null, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_2" )		; <%struct.objc_object*> [#uses=0]
	%tmp34 = load %struct.NSString** @NSXMLStoreType, align 8		; <%struct.NSString*> [#uses=1]
	%tmp37 = load %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)** getelementptr (%struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_5", i32 0, i32 0), align 8		; <%struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)*> [#uses=1]
	%tmp42 = call %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)* null( %struct.objc_object* null, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_4", i32 1 )		; <%struct.objc_object*> [#uses=1]
	%tmp45 = call %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)* %tmp37( %struct.objc_object* null, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_5", %struct.objc_object* %tmp42, %struct.NSString* null )		; <%struct.objc_object*> [#uses=1]
	%tmp48 = call %struct.objc_object* (%struct.objc_object*, %struct._message_ref_t*, ...)* null( %struct.objc_object* null, %struct._message_ref_t* @"\01L_OBJC_MESSAGE_REF_6", %struct.NSString* %tmp34, i8* null, %struct.NSURL* null, %struct.objc_object* %tmp45, %struct.NSError** null )		; <%struct.objc_object*> [#uses=0]
	unreachable
}
