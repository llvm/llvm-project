; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -regalloc=fast -optimize-regalloc=0 -relocation-model=pic | FileCheck %s

	%struct.NSError = type opaque
	%struct.NSManagedObjectContext = type opaque
	%struct.NSPersistentStoreCoordinator = type opaque
	%struct.NSString = type opaque
	%struct.NSURL = type opaque
	%struct._message_ref_t = type { ptr, ptr }
	%struct.objc_object = type {  }
	%struct.objc_selector = type opaque
@"\01L_OBJC_MESSAGE_REF_2" = external global %struct._message_ref_t		; <ptr> [#uses=1]
@"\01L_OBJC_MESSAGE_REF_6" = external global %struct._message_ref_t		; <ptr> [#uses=1]
@NSXMLStoreType = external constant ptr		; <ptr> [#uses=1]
@"\01L_OBJC_MESSAGE_REF_5" = external global %struct._message_ref_t		; <ptr> [#uses=2]
@"\01L_OBJC_MESSAGE_REF_4" = external global %struct._message_ref_t		; <ptr> [#uses=1]

; TODO: KB: ORiginal test case was just checking it compiles; is this worth keeping?
; CHECK: managedObjectContextWithModelURL
; CHECK-NOT: blr
; CHECK: .cfi_endproc

define ptr @"+[ListGenerator(Private) managedObjectContextWithModelURL:storeURL:]"(ptr %self, ptr %_cmd, ptr %modelURL, ptr %storeURL) {
entry:
	%storeCoordinator = alloca ptr		; <ptr> [#uses=0]
	%tmp29 = call ptr (ptr, ptr, ...) null( ptr null, ptr @"\01L_OBJC_MESSAGE_REF_2" )		; <ptr> [#uses=0]
	%tmp34 = load ptr, ptr @NSXMLStoreType, align 8		; <ptr> [#uses=1]
	%tmp37 = load ptr, ptr @"\01L_OBJC_MESSAGE_REF_5", align 8		; <ptr> [#uses=1]
	%tmp42 = call ptr (ptr, ptr, ...) null( ptr null, ptr @"\01L_OBJC_MESSAGE_REF_4", i32 1 )		; <ptr> [#uses=1]
	%tmp45 = call ptr (ptr, ptr, ...) %tmp37( ptr null, ptr @"\01L_OBJC_MESSAGE_REF_5", ptr %tmp42, ptr null )		; <ptr> [#uses=1]
	%tmp48 = call ptr (ptr, ptr, ...) null( ptr null, ptr @"\01L_OBJC_MESSAGE_REF_6", ptr %tmp34, ptr null, ptr null, ptr %tmp45, ptr null )		; <ptr> [#uses=0]
	unreachable
}
