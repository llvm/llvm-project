; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -regalloc=fast -optimize-regalloc=0 -relocation-model=pic | FileCheck %s

	%struct.NSError = type opaque
	%struct.NSManagedObjectContext = type opaque
	%struct.NSString = type opaque
	%struct.NSURL = type opaque
	%struct._message_ref_t = type { ptr, ptr }
	%struct.objc_object = type {  }
	%struct.objc_selector = type opaque
@"\01L_OBJC_MESSAGE_REF_2" = external global %struct._message_ref_t		; <ptr> [#uses=2]
@"\01L_OBJC_MESSAGE_REF_6" = external global %struct._message_ref_t		; <ptr> [#uses=2]
@NSXMLStoreType = external constant ptr		; <ptr> [#uses=1]
@"\01L_OBJC_MESSAGE_REF_4" = external global %struct._message_ref_t		; <ptr> [#uses=2]

; TODO: KB: ORiginal test case was just checking it compiles; is this worth keeping?
; CHECK: managedObjectContextWithModelURL
; CHECK-NOT: blr
; CHECK: .cfi_endproc

define ptr @"+[ListGenerator(Private) managedObjectContextWithModelURL:storeURL:]"(ptr %self, ptr %_cmd, ptr %modelURL, ptr %storeURL) {
entry:
	%tmp27 = load ptr, ptr @"\01L_OBJC_MESSAGE_REF_2", align 8		; <ptr> [#uses=1]
	%tmp29 = call ptr (ptr, ptr, ...) %tmp27( ptr null, ptr @"\01L_OBJC_MESSAGE_REF_2" )		; <ptr> [#uses=0]
	%tmp33 = load ptr, ptr @"\01L_OBJC_MESSAGE_REF_6", align 8		; <ptr> [#uses=1]
	%tmp34 = load ptr, ptr @NSXMLStoreType, align 8		; <ptr> [#uses=1]
	%tmp40 = load ptr, ptr @"\01L_OBJC_MESSAGE_REF_4", align 8		; <ptr> [#uses=1]
	%tmp42 = call ptr (ptr, ptr, ...) %tmp40( ptr null, ptr @"\01L_OBJC_MESSAGE_REF_4", i32 1 )		; <ptr> [#uses=0]
	%tmp48 = call ptr (ptr, ptr, ...) %tmp33( ptr null, ptr @"\01L_OBJC_MESSAGE_REF_6", ptr %tmp34, ptr null, ptr null, ptr null, ptr null )		; <ptr> [#uses=0]
	unreachable
}
