; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | grep "__trampoline_setup"

module asm "\09.lazy_reference .objc_class_name_NSImageRep"
module asm "\09.objc_class_name_NSBitmapImageRep=0"
module asm "\09.globl .objc_class_name_NSBitmapImageRep"
	%struct.CGImage = type opaque
	%"struct.FRAME.-[NSBitmapImageRep copyWithZone:]" = type { %struct.NSBitmapImageRep*, void (%struct.__block_1*, %struct.CGImage*)* }
	%struct.NSBitmapImageRep = type { %struct.NSImageRep }
	%struct.NSImageRep = type {  }
	%struct.NSZone = type opaque
	%struct.__block_1 = type { %struct.__invoke_impl, %struct.NSZone*, %struct.NSBitmapImageRep** }
	%struct.__builtin_trampoline = type { [40 x i8] }
	%struct.__invoke_impl = type { i8*, i32, i32, i8* }
	%struct._objc__method_prototype_list = type opaque
	%struct._objc_class = type { %struct._objc_class*, %struct._objc_class*, i8*, i32, i32, i32, %struct._objc_ivar_list*, %struct._objc_method_list*, %struct.objc_cache*, %struct._objc_protocol**, i8*, %struct._objc_class_ext* }
	%struct._objc_class_ext = type opaque
	%struct._objc_ivar_list = type opaque
	%struct._objc_method = type { %struct.objc_selector*, i8*, i8* }
	%struct._objc_method_list = type opaque
	%struct._objc_module = type { i32, i32, i8*, %struct._objc_symtab* }
	%struct._objc_protocol = type { %struct._objc_protocol_extension*, i8*, %struct._objc_protocol**, %struct._objc__method_prototype_list*, %struct._objc__method_prototype_list* }
	%struct._objc_protocol_extension = type opaque
	%struct._objc_super = type { %struct.objc_object*, %struct._objc_class* }
	%struct._objc_symtab = type { i32, %struct.objc_selector**, i16, i16, [1 x i8*] }
	%struct.anon = type { %struct._objc__method_prototype_list*, i32, [1 x %struct._objc_method] }
	%struct.objc_cache = type opaque
	%struct.objc_object = type opaque
	%struct.objc_selector = type opaque
	%struct.objc_super = type opaque
@_NSConcreteStackBlock = external global i8*		; <i8**> [#uses=1]
@"\01L_OBJC_SELECTOR_REFERENCES_1" = internal global %struct.objc_selector* bitcast ([34 x i8]* @"\01L_OBJC_METH_VAR_NAME_1" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip"		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_CLASS_NSBitmapImageRep" = internal global %struct._objc_class { %struct._objc_class* @"\01L_OBJC_METACLASS_NSBitmapImageRep", %struct._objc_class* bitcast ([11 x i8]* @"\01L_OBJC_CLASS_NAME_1" to %struct._objc_class*), i8* getelementptr ([17 x i8], [17 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), i32 0, i32 1, i32 0, %struct._objc_ivar_list* null, %struct._objc_method_list* bitcast ({ i8*, i32, [1 x %struct._objc_method] }* @"\01L_OBJC_INSTANCE_METHODS_NSBitmapImageRep" to %struct._objc_method_list*), %struct.objc_cache* null, %struct._objc_protocol** null, i8* null, %struct._objc_class_ext* null }, section "__OBJC,__class,regular,no_dead_strip"		; <%struct._objc_class*> [#uses=3]
@"\01L_OBJC_SELECTOR_REFERENCES_0" = internal global %struct.objc_selector* bitcast ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_0" to %struct.objc_selector*), section "__OBJC,__message_refs,literal_pointers,no_dead_strip"		; <%struct.objc_selector**> [#uses=2]
@"\01L_OBJC_SYMBOLS" = internal global { i32, %struct.objc_selector**, i16, i16, [1 x %struct._objc_class*] } { i32 0, %struct.objc_selector** null, i16 1, i16 0, [1 x %struct._objc_class*] [ %struct._objc_class* @"\01L_OBJC_CLASS_NSBitmapImageRep" ] }, section "__OBJC,__symbols,regular,no_dead_strip"		; <{ i32, %struct.objc_selector**, i16, i16, [1 x %struct._objc_class*] }*> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_0" = internal global [14 x i8] c"copyWithZone:\00", section "__TEXT,__cstring,cstring_literals", align 4		; <[14 x i8]*> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_0" = internal global [20 x i8] c"@12@0:4^{_NSZone=}8\00", section "__TEXT,__cstring,cstring_literals", align 4		; <[20 x i8]*> [#uses=1]
@"\01L_OBJC_INSTANCE_METHODS_NSBitmapImageRep" = internal global { i8*, i32, [1 x %struct._objc_method] } { i8* null, i32 1, [1 x %struct._objc_method] [ %struct._objc_method { %struct.objc_selector* bitcast ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_0" to %struct.objc_selector*), i8* getelementptr ([20 x i8], [20 x i8]* @"\01L_OBJC_METH_VAR_TYPE_0", i32 0, i32 0), i8* bitcast (%struct.objc_object* (%struct.NSBitmapImageRep*, %struct.objc_selector*, %struct.NSZone*)* @"-[NSBitmapImageRep copyWithZone:]" to i8*) } ] }, section "__OBJC,__inst_meth,regular,no_dead_strip"		; <{ i8*, i32, [1 x %struct._objc_method] }*> [#uses=2]
@"\01L_OBJC_CLASS_NAME_0" = internal global [17 x i8] c"NSBitmapImageRep\00", section "__TEXT,__cstring,cstring_literals", align 4		; <[17 x i8]*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_1" = internal global [11 x i8] c"NSImageRep\00", section "__TEXT,__cstring,cstring_literals", align 4		; <[11 x i8]*> [#uses=2]
@"\01L_OBJC_METACLASS_NSBitmapImageRep" = internal global %struct._objc_class { %struct._objc_class* bitcast ([11 x i8]* @"\01L_OBJC_CLASS_NAME_1" to %struct._objc_class*), %struct._objc_class* bitcast ([11 x i8]* @"\01L_OBJC_CLASS_NAME_1" to %struct._objc_class*), i8* getelementptr ([17 x i8], [17 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), i32 0, i32 2, i32 48, %struct._objc_ivar_list* null, %struct._objc_method_list* null, %struct.objc_cache* null, %struct._objc_protocol** null, i8* null, %struct._objc_class_ext* null }, section "__OBJC,__meta_class,regular,no_dead_strip"		; <%struct._objc_class*> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_1" = internal global [34 x i8] c"_performBlockUsingBackingCGImage:\00", section "__TEXT,__cstring,cstring_literals", align 4		; <[34 x i8]*> [#uses=2]
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] zeroinitializer, section "__OBJC, __image_info,regular"		; <[2 x i32]*> [#uses=1]
@"\01L_OBJC_CLASS_NAME_2" = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals", align 4		; <[1 x i8]*> [#uses=1]
@"\01L_OBJC_MODULES" = internal global %struct._objc_module { i32 7, i32 16, i8* getelementptr ([1 x i8], [1 x i8]* @"\01L_OBJC_CLASS_NAME_2", i32 0, i32 0), %struct._objc_symtab* bitcast ({ i32, %struct.objc_selector**, i16, i16, [1 x %struct._objc_class*] }* @"\01L_OBJC_SYMBOLS" to %struct._objc_symtab*) }, section "__OBJC,__module_info,regular,no_dead_strip"		; <%struct._objc_module*> [#uses=1]
@llvm.used = appending global [14 x i8*] [ i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_1" to i8*), i8* bitcast (%struct._objc_class* @"\01L_OBJC_CLASS_NSBitmapImageRep" to i8*), i8* bitcast (%struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_0" to i8*), i8* bitcast ({ i32, %struct.objc_selector**, i16, i16, [1 x %struct._objc_class*] }* @"\01L_OBJC_SYMBOLS" to i8*), i8* getelementptr ([14 x i8], [14 x i8]* @"\01L_OBJC_METH_VAR_NAME_0", i32 0, i32 0), i8* getelementptr ([20 x i8], [20 x i8]* @"\01L_OBJC_METH_VAR_TYPE_0", i32 0, i32 0), i8* bitcast ({ i8*, i32, [1 x %struct._objc_method] }* @"\01L_OBJC_INSTANCE_METHODS_NSBitmapImageRep" to i8*), i8* getelementptr ([17 x i8], [17 x i8]* @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 0), i8* getelementptr ([11 x i8], [11 x i8]* @"\01L_OBJC_CLASS_NAME_1", i32 0, i32 0), i8* bitcast (%struct._objc_class* @"\01L_OBJC_METACLASS_NSBitmapImageRep" to i8*), i8* getelementptr ([34 x i8], [34 x i8]* @"\01L_OBJC_METH_VAR_NAME_1", i32 0, i32 0), i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*), i8* getelementptr ([1 x i8], [1 x i8]* @"\01L_OBJC_CLASS_NAME_2", i32 0, i32 0), i8* bitcast (%struct._objc_module* @"\01L_OBJC_MODULES" to i8*) ], section "llvm.metadata"		; <[14 x i8*]*> [#uses=0]

define internal %struct.objc_object* @"-[NSBitmapImageRep copyWithZone:]"(%struct.NSBitmapImageRep* %self, %struct.objc_selector* %_cmd, %struct.NSZone* %zone) nounwind {
entry:
	%self_addr = alloca %struct.NSBitmapImageRep*		; <%struct.NSBitmapImageRep**> [#uses=2]
	%_cmd_addr = alloca %struct.objc_selector*		; <%struct.objc_selector**> [#uses=1]
	%zone_addr = alloca %struct.NSZone*		; <%struct.NSZone**> [#uses=2]
	%retval = alloca %struct.objc_object*		; <%struct.objc_object**> [#uses=1]
	%__block_holder_tmp_1.0 = alloca %struct.__block_1		; <%struct.__block_1*> [#uses=7]
	%new = alloca %struct.NSBitmapImageRep*		; <%struct.NSBitmapImageRep**> [#uses=2]
	%self.1 = alloca %struct.objc_object*		; <%struct.objc_object**> [#uses=2]
	%0 = alloca i8*		; <i8**> [#uses=2]
	%TRAMP.9 = alloca %struct.__builtin_trampoline, align 4		; <%struct.__builtin_trampoline*> [#uses=1]
	%1 = alloca void (%struct.__block_1*, %struct.CGImage*)*		; <void (%struct.__block_1*, %struct.CGImage*)**> [#uses=2]
	%2 = alloca %struct.NSBitmapImageRep*		; <%struct.NSBitmapImageRep**> [#uses=2]
	%FRAME.7 = alloca %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"		; <%"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"*> [#uses=5]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store %struct.NSBitmapImageRep* %self, %struct.NSBitmapImageRep** %self_addr
	store %struct.objc_selector* %_cmd, %struct.objc_selector** %_cmd_addr
	store %struct.NSZone* %zone, %struct.NSZone** %zone_addr
	%3 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"* %FRAME.7, i32 0, i32 0		; <%struct.NSBitmapImageRep**> [#uses=1]
	%4 = load %struct.NSBitmapImageRep*, %struct.NSBitmapImageRep** %self_addr, align 4		; <%struct.NSBitmapImageRep*> [#uses=1]
	store %struct.NSBitmapImageRep* %4, %struct.NSBitmapImageRep** %3, align 4
	%TRAMP.91 = bitcast %struct.__builtin_trampoline* %TRAMP.9 to i8*		; <i8*> [#uses=1]
	%FRAME.72 = bitcast %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"* %FRAME.7 to i8*		; <i8*> [#uses=1]
	call void @llvm.init.trampoline(i8* %TRAMP.91, i8* bitcast (void (%"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"*, %struct.__block_1*, %struct.CGImage*)* @__helper_1.1632 to i8*), i8* %FRAME.72)		; <i8*> [#uses=1]
        %tramp = call i8* @llvm.adjust.trampoline(i8* %TRAMP.91)
	store i8* %tramp, i8** %0, align 4
	%5 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"* %FRAME.7, i32 0, i32 1		; <void (%struct.__block_1*, %struct.CGImage*)**> [#uses=1]
	%6 = load i8*, i8** %0, align 4		; <i8*> [#uses=1]
	%7 = bitcast i8* %6 to void (%struct.__block_1*, %struct.CGImage*)*		; <void (%struct.__block_1*, %struct.CGImage*)*> [#uses=1]
	store void (%struct.__block_1*, %struct.CGImage*)* %7, void (%struct.__block_1*, %struct.CGImage*)** %5, align 4
	store %struct.NSBitmapImageRep* null, %struct.NSBitmapImageRep** %new, align 4
	%8 = getelementptr %struct.__block_1, %struct.__block_1* %__block_holder_tmp_1.0, i32 0, i32 0		; <%struct.__invoke_impl*> [#uses=1]
	%9 = getelementptr %struct.__invoke_impl, %struct.__invoke_impl* %8, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* bitcast (i8** @_NSConcreteStackBlock to i8*), i8** %9, align 4
	%10 = getelementptr %struct.__block_1, %struct.__block_1* %__block_holder_tmp_1.0, i32 0, i32 0		; <%struct.__invoke_impl*> [#uses=1]
	%11 = getelementptr %struct.__invoke_impl, %struct.__invoke_impl* %10, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 67108864, i32* %11, align 4
	%12 = getelementptr %struct.__block_1, %struct.__block_1* %__block_holder_tmp_1.0, i32 0, i32 0		; <%struct.__invoke_impl*> [#uses=1]
	%13 = getelementptr %struct.__invoke_impl, %struct.__invoke_impl* %12, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 24, i32* %13, align 4
	%14 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"* %FRAME.7, i32 0, i32 1		; <void (%struct.__block_1*, %struct.CGImage*)**> [#uses=1]
	%15 = load void (%struct.__block_1*, %struct.CGImage*)*, void (%struct.__block_1*, %struct.CGImage*)** %14, align 4		; <void (%struct.__block_1*, %struct.CGImage*)*> [#uses=1]
	store void (%struct.__block_1*, %struct.CGImage*)* %15, void (%struct.__block_1*, %struct.CGImage*)** %1, align 4
	%16 = getelementptr %struct.__block_1, %struct.__block_1* %__block_holder_tmp_1.0, i32 0, i32 0		; <%struct.__invoke_impl*> [#uses=1]
	%17 = getelementptr %struct.__invoke_impl, %struct.__invoke_impl* %16, i32 0, i32 3		; <i8**> [#uses=1]
	%18 = load void (%struct.__block_1*, %struct.CGImage*)*, void (%struct.__block_1*, %struct.CGImage*)** %1, align 4		; <void (%struct.__block_1*, %struct.CGImage*)*> [#uses=1]
	%19 = bitcast void (%struct.__block_1*, %struct.CGImage*)* %18 to i8*		; <i8*> [#uses=1]
	store i8* %19, i8** %17, align 4
	%20 = getelementptr %struct.__block_1, %struct.__block_1* %__block_holder_tmp_1.0, i32 0, i32 1		; <%struct.NSZone**> [#uses=1]
	%21 = load %struct.NSZone*, %struct.NSZone** %zone_addr, align 4		; <%struct.NSZone*> [#uses=1]
	store %struct.NSZone* %21, %struct.NSZone** %20, align 4
	%22 = getelementptr %struct.__block_1, %struct.__block_1* %__block_holder_tmp_1.0, i32 0, i32 2		; <%struct.NSBitmapImageRep***> [#uses=1]
	store %struct.NSBitmapImageRep** %new, %struct.NSBitmapImageRep*** %22, align 4
	%23 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"* %FRAME.7, i32 0, i32 0		; <%struct.NSBitmapImageRep**> [#uses=1]
	%24 = load %struct.NSBitmapImageRep*, %struct.NSBitmapImageRep** %23, align 4		; <%struct.NSBitmapImageRep*> [#uses=1]
	store %struct.NSBitmapImageRep* %24, %struct.NSBitmapImageRep** %2, align 4
	%25 = load %struct.NSBitmapImageRep*, %struct.NSBitmapImageRep** %2, align 4		; <%struct.NSBitmapImageRep*> [#uses=1]
	%26 = bitcast %struct.NSBitmapImageRep* %25 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	store %struct.objc_object* %26, %struct.objc_object** %self.1, align 4
	%27 = load %struct.objc_selector*, %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_1", align 4		; <%struct.objc_selector*> [#uses=1]
	%__block_holder_tmp_1.03 = bitcast %struct.__block_1* %__block_holder_tmp_1.0 to void (%struct.CGImage*)*		; <void (%struct.CGImage*)*> [#uses=1]
	%28 = load %struct.objc_object*, %struct.objc_object** %self.1, align 4		; <%struct.objc_object*> [#uses=1]
	%29 = call %struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...) inttoptr (i64 4294901504 to %struct.objc_object* (%struct.objc_object*, %struct.objc_selector*, ...)*)(%struct.objc_object* %28, %struct.objc_selector* %27, void (%struct.CGImage*)* %__block_holder_tmp_1.03) nounwind		; <%struct.objc_object*> [#uses=0]
	br label %return

return:		; preds = %entry
	%retval5 = load %struct.objc_object*, %struct.objc_object** %retval		; <%struct.objc_object*> [#uses=1]
	ret %struct.objc_object* %retval5
}

declare void @llvm.init.trampoline(i8*, i8*, i8*) nounwind
declare i8* @llvm.adjust.trampoline(i8*) nounwind

define internal void @__helper_1.1632(%"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"* nest %CHAIN.8, %struct.__block_1* %_self, %struct.CGImage* %cgImage) nounwind {
entry:
	%CHAIN.8_addr = alloca %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"*		; <%"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"**> [#uses=2]
	%_self_addr = alloca %struct.__block_1*		; <%struct.__block_1**> [#uses=3]
	%cgImage_addr = alloca %struct.CGImage*		; <%struct.CGImage**> [#uses=1]
	%zone = alloca %struct.NSZone*		; <%struct.NSZone**> [#uses=2]
	%objc_super = alloca %struct._objc_super		; <%struct._objc_super*> [#uses=3]
	%new = alloca %struct.NSBitmapImageRep**		; <%struct.NSBitmapImageRep***> [#uses=2]
	%objc_super.5 = alloca %struct.objc_super*		; <%struct.objc_super**> [#uses=2]
	%0 = alloca %struct.NSBitmapImageRep*		; <%struct.NSBitmapImageRep**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"* %CHAIN.8, %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"** %CHAIN.8_addr
	store %struct.__block_1* %_self, %struct.__block_1** %_self_addr
	store %struct.CGImage* %cgImage, %struct.CGImage** %cgImage_addr
	%1 = load %struct.__block_1*, %struct.__block_1** %_self_addr, align 4		; <%struct.__block_1*> [#uses=1]
	%2 = getelementptr %struct.__block_1, %struct.__block_1* %1, i32 0, i32 2		; <%struct.NSBitmapImageRep***> [#uses=1]
	%3 = load %struct.NSBitmapImageRep**, %struct.NSBitmapImageRep*** %2, align 4		; <%struct.NSBitmapImageRep**> [#uses=1]
	store %struct.NSBitmapImageRep** %3, %struct.NSBitmapImageRep*** %new, align 4
	%4 = load %struct.__block_1*, %struct.__block_1** %_self_addr, align 4		; <%struct.__block_1*> [#uses=1]
	%5 = getelementptr %struct.__block_1, %struct.__block_1* %4, i32 0, i32 1		; <%struct.NSZone**> [#uses=1]
	%6 = load %struct.NSZone*, %struct.NSZone** %5, align 4		; <%struct.NSZone*> [#uses=1]
	store %struct.NSZone* %6, %struct.NSZone** %zone, align 4
	%7 = load %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"*, %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"** %CHAIN.8_addr, align 4		; <%"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"*> [#uses=1]
	%8 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"* %7, i32 0, i32 0		; <%struct.NSBitmapImageRep**> [#uses=1]
	%9 = load %struct.NSBitmapImageRep*, %struct.NSBitmapImageRep** %8, align 4		; <%struct.NSBitmapImageRep*> [#uses=1]
	store %struct.NSBitmapImageRep* %9, %struct.NSBitmapImageRep** %0, align 4
	%10 = load %struct.NSBitmapImageRep*, %struct.NSBitmapImageRep** %0, align 4		; <%struct.NSBitmapImageRep*> [#uses=1]
	%11 = bitcast %struct.NSBitmapImageRep* %10 to %struct.objc_object*		; <%struct.objc_object*> [#uses=1]
	%12 = getelementptr %struct._objc_super, %struct._objc_super* %objc_super, i32 0, i32 0		; <%struct.objc_object**> [#uses=1]
	store %struct.objc_object* %11, %struct.objc_object** %12, align 4
	%13 = load %struct._objc_class*, %struct._objc_class** getelementptr (%struct._objc_class, %struct._objc_class* @"\01L_OBJC_CLASS_NSBitmapImageRep", i32 0, i32 1), align 4		; <%struct._objc_class*> [#uses=1]
	%14 = getelementptr %struct._objc_super, %struct._objc_super* %objc_super, i32 0, i32 1		; <%struct._objc_class**> [#uses=1]
	store %struct._objc_class* %13, %struct._objc_class** %14, align 4
	%objc_super1 = bitcast %struct._objc_super* %objc_super to %struct.objc_super*		; <%struct.objc_super*> [#uses=1]
	store %struct.objc_super* %objc_super1, %struct.objc_super** %objc_super.5, align 4
	%15 = load %struct.objc_selector*, %struct.objc_selector** @"\01L_OBJC_SELECTOR_REFERENCES_0", align 4		; <%struct.objc_selector*> [#uses=1]
	%16 = load %struct.objc_super*, %struct.objc_super** %objc_super.5, align 4		; <%struct.objc_super*> [#uses=1]
	%17 = load %struct.NSZone*, %struct.NSZone** %zone, align 4		; <%struct.NSZone*> [#uses=1]
	%18 = call %struct.objc_object* (%struct.objc_super*, %struct.objc_selector*, ...) @objc_msgSendSuper(%struct.objc_super* %16, %struct.objc_selector* %15, %struct.NSZone* %17) nounwind		; <%struct.objc_object*> [#uses=1]
	%19 = bitcast %struct.objc_object* %18 to %struct.NSBitmapImageRep*		; <%struct.NSBitmapImageRep*> [#uses=1]
	%20 = load %struct.NSBitmapImageRep**, %struct.NSBitmapImageRep*** %new, align 4		; <%struct.NSBitmapImageRep**> [#uses=1]
	store %struct.NSBitmapImageRep* %19, %struct.NSBitmapImageRep** %20, align 4
	br label %return

return:		; preds = %entry
	ret void
}

declare %struct.objc_object* @objc_msgSendSuper(%struct.objc_super*, %struct.objc_selector*, ...)
