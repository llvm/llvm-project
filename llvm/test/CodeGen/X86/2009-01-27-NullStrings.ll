; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin | grep {\\.cstring} | count 1
	%struct.A = type {  }
	%struct.NSString = type opaque
	%struct.__builtin_CFString = type { i32*, i32, i8*, i32 }
	%struct._objc_module = type { i32, i32, i8*, %struct._objc_symtab* }
	%struct._objc_symtab = type { i32, %struct.objc_selector**, i16, i16 }
	%struct.objc_object = type opaque
	%struct.objc_selector = type opaque
@"\01L_unnamed_cfstring_0" = internal constant %struct.__builtin_CFString { i32* getelementptr ([0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr ([1 x i8]* @"\01LC", i32 0, i32 0), i32 0 }, section "__DATA, __cfstring"		; <%struct.__builtin_CFString*> [#uses=1]
@__CFConstantStringClassReference = external global [0 x i32]		; <[0 x i32]*> [#uses=1]
@"\01LC" = internal constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=1]
@"\01L_OBJC_SELECTOR_REFERENCES_0" = internal global %struct.objc_selector* bitcast ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_0" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_SYMBOLS" = internal global %struct._objc_symtab zeroinitializer, section "__OBJC,__symbols,regular,no_dead_strip", align 4		; <%struct._objc_symtab*> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_0" = internal global [6 x i8] c"bork:\00", section "__TEXT,__cstring,cstring_literals", align 1		; <[6 x i8]*> [#uses=2]
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] zeroinitializer, section "__OBJC, __image_info,regular"		; <[2 x i32]*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_0" = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals", align 1		; <[1 x i8]*> [#uses=1]
@"\01L_OBJC_MODULES" = internal global %struct._objc_module { i32 7, i32 16, i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), %struct._objc_symtab* @"\01L_OBJC_SYMBOLS" }, section "__OBJC,__module_info,regular,no_dead_strip", align 4		; <%struct._objc_module*> [#uses=1]
@llvm.used = appending global [6 x i8*] [ i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_0" to i8*), i8* bitcast (%struct._objc_symtab* @"\01L_OBJC_SYMBOLS" to i8*), i8* getelementptr ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_0", i32 0, i32 0), i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*), i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), i8* bitcast (%struct._objc_module* @"\01L_OBJC_MODULES" to i8*) ], section "llvm.metadata"		; <[6 x i8*]*> [#uses=0]

define void @func(%struct.A* %a) nounwind {
entry:
	%a_addr = alloca %struct.A*		; <%struct.A**> [#uses=2]
	%a.0 = alloca %struct.objc_object*		; <%struct.objc_object**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store %struct.A* %a, %struct.A** %a_addr
	%0 = load %struct.A** %a_addr, align 4		; <%struct.A*> [#uses=1]
	%1 = bitcast %struct.A* %0 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	store %struct.objc_object* %1, %struct.objc_object** %a.0, align 4
	%2 = load %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_0", align 4		; <%struct.objc_selector*> [#uses=1]
	%3 = load %struct.objc_object** %a.0, align 4		; <%struct.objc_object*> [#uses=1]
	call void bitcast (%struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)* @objc_msgSend to void (%struct.objc_object*, %struct.objc_selector*, %struct.NSString*)*)(%struct.objc_object* %3, %struct.objc_selector* %2, %struct.NSString* bitcast (%struct.__builtin_CFString* @"\01L_unnamed_cfstring_0" to %struct.NSString*)) nounwind
	br label %return

return:		; preds = %entry
	ret void
}

declare %struct.objc_object* @objc_msgSend(%struct.objc_object*, %struct.objc_selector*, ...)
