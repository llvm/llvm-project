; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-apple-darwin -enable-eh  | grep {isNullOrNil].eh"} | count 2

	%struct.NSString = type {  }
	%struct._objc__method_prototype_list = type opaque
	%struct._objc_category = type { i8*, i8*, %struct._objc_method_list*, %struct._objc_method_list*, %struct._objc_protocol**, i32, %struct._prop_list_t* }
	%struct._objc_method = type { %struct.objc_selector*, i8*, i8* }
	%struct._objc_method_list = type opaque
	%struct._objc_module = type { i32, i32, i8*, %struct._objc_symtab* }
	%struct._objc_protocol = type { %struct._objc_protocol_extension*, i8*, %struct._objc_protocol**, %struct._objc__method_prototype_list*, %struct._objc__method_prototype_list* }
	%struct._objc_protocol_extension = type opaque
	%struct._objc_symtab = type { i32, %struct.objc_selector**, i16, i16, [1 x i8*] }
	%struct._prop_list_t = type opaque
	%struct.anon = type { %struct._objc__method_prototype_list*, i32, [1 x %struct._objc_method] }
	%struct.objc_selector = type opaque
@"\01L_OBJC_SYMBOLS" = internal global { i32, i32, i16, i16, [1 x %struct._objc_category*] } {
    i32 0, 
    i32 0, 
    i16 0, 
    i16 1, 
    [1 x %struct._objc_category*] [ %struct._objc_category* bitcast ({ i8*, i8*, %struct._objc_method_list*, i32, i32, i32, i32 }* @"\01L_OBJC_CATEGORY_NSString_local" to %struct._objc_category*) ] }, section "__OBJC,__symbols,regular,no_dead_strip"		; <{ i32, i32, i16, i16, [1 x %struct._objc_category*] }*> [#uses=2]
@"\01L_OBJC_CATEGORY_INSTANCE_METHODS_NSString_local" = internal global { i32, i32, [1 x %struct._objc_method] } {
    i32 0, 
    i32 1, 
    [1 x %struct._objc_method] [ %struct._objc_method {
        %struct.objc_selector* bitcast ([12 x i8]* @"\01L_OBJC_METH_VAR_NAME_0" to %struct.objc_selector*), 
        i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_0", i32 0, i32 0), 
        i8* bitcast (i8 (%struct.NSString*, %struct.objc_selector*) signext * @"-[NSString(local) isNullOrNil]" to i8*) } ] }, section "__OBJC,__cat_inst_meth,regular,no_dead_strip"		; <{ i32, i32, [1 x %struct._objc_method] }*> [#uses=3]
@"\01L_OBJC_CATEGORY_NSString_local" = internal global { i8*, i8*, %struct._objc_method_list*, i32, i32, i32, i32 } {
    i8* getelementptr ([6 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), 
    i8* getelementptr ([9 x i8]* @"\01L_OBJC_CLASS_NAME_1", i32 0, i32 0), 
    %struct._objc_method_list* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01L_OBJC_CATEGORY_INSTANCE_METHODS_NSString_local" to %struct._objc_method_list*), 
    i32 0, 
    i32 0, 
    i32 28, 
    i32 0 }, section "__OBJC,__category,regular,no_dead_strip"		; <{ i8*, i8*, %struct._objc_method_list*, i32, i32, i32, i32 }*> [#uses=2]
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] zeroinitializer, section "__OBJC,__image_info,regular"		; <[2 x i32]*> [#uses=1]
@"\01L_OBJC_MODULES" = internal global %struct._objc_module {
    i32 7, 
    i32 16, 
    i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_2", i32 0, i32 0), 
    %struct._objc_symtab* bitcast ({ i32, i32, i16, i16, [1 x %struct._objc_category*] }* @"\01L_OBJC_SYMBOLS" to %struct._objc_symtab*) }, section "__OBJC,__module_info,regular,no_dead_strip"		; <%struct._objc_module*> [#uses=1]
@"\01.objc_class_ref_NSString" = internal global i8* @"\01.objc_class_name_NSString"		; <i8**> [#uses=0]
@"\01.objc_class_name_NSString" = external global i8		; <i8*> [#uses=1]
@"\01.objc_category_name_NSString_local" = constant i32 0		; <i32*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_2" = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals"		; <[1 x i8]*> [#uses=2]
@"\01L_OBJC_CLASS_NAME_1" = internal global [9 x i8] c"NSString\00", section "__TEXT,__cstring,cstring_literals"		; <[9 x i8]*> [#uses=2]
@"\01L_OBJC_CLASS_NAME_0" = internal global [6 x i8] c"local\00", section "__TEXT,__cstring,cstring_literals"		; <[6 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_0" = internal global [12 x i8] c"isNullOrNil\00", section "__TEXT,__cstring,cstring_literals"		; <[12 x i8]*> [#uses=3]
@"\01L_OBJC_METH_VAR_TYPE_0" = internal global [7 x i8] c"c8@0:4\00", section "__TEXT,__cstring,cstring_literals"		; <[7 x i8]*> [#uses=2]
@llvm.used = appending global [11 x i8*] [ i8* bitcast ({ i32, i32, i16, i16, [1 x %struct._objc_category*] }* @"\01L_OBJC_SYMBOLS" to i8*), i8* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01L_OBJC_CATEGORY_INSTANCE_METHODS_NSString_local" to i8*), i8* bitcast ({ i8*, i8*, %struct._objc_method_list*, i32, i32, i32, i32 }* @"\01L_OBJC_CATEGORY_NSString_local" to i8*), i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*), i8* bitcast (%struct._objc_module* @"\01L_OBJC_MODULES" to i8*), i8* bitcast (i32* @"\01.objc_category_name_NSString_local" to i8*), i8* getelementptr ([1 x i8]* @"\01L_OBJC_CLASS_NAME_2", i32 0, i32 0), i8* getelementptr ([9 x i8]* @"\01L_OBJC_CLASS_NAME_1", i32 0, i32 0), i8* getelementptr ([6 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), i8* getelementptr ([12 x i8]* @"\01L_OBJC_METH_VAR_NAME_0", i32 0, i32 0), i8* getelementptr ([7 x i8]* @"\01L_OBJC_METH_VAR_TYPE_0", i32 0, i32 0) ], section "llvm.metadata"		; <[11 x i8*]*> [#uses=0]

define internal i8 @"-[NSString(local) isNullOrNil]"(%struct.NSString* %self, %struct.objc_selector* %_cmd) signext  {
entry:
	%self_addr = alloca %struct.NSString*		; <%struct.NSString**> [#uses=1]
	%_cmd_addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=1]
	%retval = alloca i8, align 1		; <i8*> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store %struct.NSString* %self, %struct.NSString** %self_addr
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd_addr
	br label %return

return:		; preds = %entry
	%retval1 = load i8* %retval		; <i8> [#uses=1]
	ret i8 %retval1
}
