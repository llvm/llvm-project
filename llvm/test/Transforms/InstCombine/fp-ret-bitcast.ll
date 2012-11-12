; RUN: opt < %s -instcombine -S | \
; RUN:    grep "call float bitcast" | count 1
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"
	%struct.NSObject = type { %struct.objc_class* }
 	%struct.NSArray = type { %struct.NSObject }
	%struct.objc_class = type opaque
 	%struct.objc_selector = type opaque

@"\01L_OBJC_METH_VAR_NAME_112" = internal global [15 x i8] c"whiteComponent\00", section "__TEXT,__cstring,cstring_literals"
@"\01L_OBJC_SELECTOR_REFERENCES_81" = internal global %struct.objc_selector* bitcast ([15 x i8]* @"\01L_OBJC_METH_VAR_NAME_112" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip"

define void @bork() nounwind  {
entry:
	%color = alloca %struct.NSArray*
	%color.466 = alloca %struct.NSObject*
	%tmp103 = load %struct.NSArray** %color, align 4
	%tmp103104 = getelementptr %struct.NSArray* %tmp103, i32 0, i32 0
	store %struct.NSObject* %tmp103104, %struct.NSObject** %color.466, align 4
	%tmp105 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_81", align 4
	%tmp106 = load %struct.NSObject** %color.466, align 4
	%tmp107 = call float bitcast (void (%struct.NSObject*, ...)* @objc_msgSend_fpret to float (%struct.NSObject*, %struct.objc_selector*)*)( %struct.NSObject* %tmp106, %struct.objc_selector* %tmp105 ) nounwind
	br label %exit

exit:
	ret void
}

declare void @objc_msgSend_fpret(%struct.NSObject*, ...)
