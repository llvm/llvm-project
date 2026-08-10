; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | grep "__trampoline_setup"

module asm "\09.lazy_reference .objc_class_name_NSImageRep"
module asm "\09.objc_class_name_NSBitmapImageRep=0"
module asm "\09.globl .objc_class_name_NSBitmapImageRep"
	%struct.CGImage = type opaque
	%"struct.FRAME.-[NSBitmapImageRep copyWithZone:]" = type { ptr, ptr }
	%struct.NSBitmapImageRep = type { %struct.NSImageRep }
	%struct.NSImageRep = type {  }
	%struct.NSZone = type opaque
	%struct.__block_1 = type { %struct.__invoke_impl, ptr, ptr }
	%struct.__builtin_trampoline = type { [40 x i8] }
	%struct.__invoke_impl = type { ptr, i32, i32, ptr }
	%struct._objc__method_prototype_list = type opaque
	%struct._objc_class = type { ptr, ptr, ptr, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr }
	%struct._objc_class_ext = type opaque
	%struct._objc_ivar_list = type opaque
	%struct._objc_method = type { ptr, ptr, ptr }
	%struct._objc_method_list = type opaque
	%struct._objc_module = type { i32, i32, ptr, ptr }
	%struct._objc_protocol = type { ptr, ptr, ptr, ptr, ptr }
	%struct._objc_protocol_extension = type opaque
	%struct._objc_super = type { ptr, ptr }
	%struct._objc_symtab = type { i32, ptr, i16, i16, [1 x ptr] }
	%struct.anon = type { ptr, i32, [1 x %struct._objc_method] }
	%struct.objc_cache = type opaque
	%struct.objc_object = type opaque
	%struct.objc_selector = type opaque
	%struct.objc_super = type opaque
@_NSConcreteStackBlock = external global ptr		; <ptr> [#uses=1]
@"\01L_OBJC_SELECTOR_REFERENCES_1" = internal global ptr @"\01L_OBJC_METH_VAR_NAME_1", section "__OBJC,__message_refs,literal_pointers,no_dead_strip"		; <ptr> [#uses=2]
@"\01L_OBJC_CLASS_NSBitmapImageRep" = internal global %struct._objc_class { ptr @"\01L_OBJC_METACLASS_NSBitmapImageRep", ptr @"\01L_OBJC_CLASS_NAME_1", ptr @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 1, i32 0, ptr null, ptr @"\01L_OBJC_INSTANCE_METHODS_NSBitmapImageRep", ptr null, ptr null, ptr null, ptr null }, section "__OBJC,__class,regular,no_dead_strip"		; <ptr> [#uses=3]
@"\01L_OBJC_SELECTOR_REFERENCES_0" = internal global ptr @"\01L_OBJC_METH_VAR_NAME_0", section "__OBJC,__message_refs,literal_pointers,no_dead_strip"		; <ptr> [#uses=2]
@"\01L_OBJC_SYMBOLS" = internal global { i32, ptr, i16, i16, [1 x ptr] } { i32 0, ptr null, i16 1, i16 0, [1 x ptr] [ ptr @"\01L_OBJC_CLASS_NSBitmapImageRep" ] }, section "__OBJC,__symbols,regular,no_dead_strip"		; <ptr> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_0" = internal global [14 x i8] c"copyWithZone:\00", section "__TEXT,__cstring,cstring_literals", align 4		; <ptr> [#uses=2]
@"\01L_OBJC_METH_VAR_TYPE_0" = internal global [20 x i8] c"@12@0:4^{_NSZone=}8\00", section "__TEXT,__cstring,cstring_literals", align 4		; <ptr> [#uses=1]
@"\01L_OBJC_INSTANCE_METHODS_NSBitmapImageRep" = internal global { ptr, i32, [1 x %struct._objc_method] } { ptr null, i32 1, [1 x %struct._objc_method] [ %struct._objc_method { ptr @"\01L_OBJC_METH_VAR_NAME_0", ptr @"\01L_OBJC_METH_VAR_TYPE_0", ptr @"-[NSBitmapImageRep copyWithZone:]" } ] }, section "__OBJC,__inst_meth,regular,no_dead_strip"		; <ptr> [#uses=2]
@"\01L_OBJC_CLASS_NAME_0" = internal global [17 x i8] c"NSBitmapImageRep\00", section "__TEXT,__cstring,cstring_literals", align 4		; <ptr> [#uses=1]
@"\01L_OBJC_CLASS_NAME_1" = internal global [11 x i8] c"NSImageRep\00", section "__TEXT,__cstring,cstring_literals", align 4		; <ptr> [#uses=2]
@"\01L_OBJC_METACLASS_NSBitmapImageRep" = internal global %struct._objc_class { ptr @"\01L_OBJC_CLASS_NAME_1", ptr @"\01L_OBJC_CLASS_NAME_1", ptr @"\01L_OBJC_CLASS_NAME_0", i32 0, i32 2, i32 48, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__OBJC,__meta_class,regular,no_dead_strip"		; <ptr> [#uses=2]
@"\01L_OBJC_METH_VAR_NAME_1" = internal global [34 x i8] c"_performBlockUsingBackingCGImage:\00", section "__TEXT,__cstring,cstring_literals", align 4		; <ptr> [#uses=2]
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] zeroinitializer, section "__OBJC, __image_info,regular"		; <ptr> [#uses=1]
@"\01L_OBJC_CLASS_NAME_2" = internal global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals", align 4		; <ptr> [#uses=1]
@"\01L_OBJC_MODULES" = internal global %struct._objc_module { i32 7, i32 16, ptr @"\01L_OBJC_CLASS_NAME_2", ptr @"\01L_OBJC_SYMBOLS" }, section "__OBJC,__module_info,regular,no_dead_strip"		; <ptr> [#uses=1]
@llvm.used = appending global [14 x ptr] [ ptr @"\01L_OBJC_SELECTOR_REFERENCES_1", ptr @"\01L_OBJC_CLASS_NSBitmapImageRep", ptr @"\01L_OBJC_SELECTOR_REFERENCES_0", ptr @"\01L_OBJC_SYMBOLS", ptr @"\01L_OBJC_METH_VAR_NAME_0", ptr @"\01L_OBJC_METH_VAR_TYPE_0", ptr @"\01L_OBJC_INSTANCE_METHODS_NSBitmapImageRep", ptr @"\01L_OBJC_CLASS_NAME_0", ptr @"\01L_OBJC_CLASS_NAME_1", ptr @"\01L_OBJC_METACLASS_NSBitmapImageRep", ptr @"\01L_OBJC_METH_VAR_NAME_1", ptr @"\01L_OBJC_IMAGE_INFO", ptr @"\01L_OBJC_CLASS_NAME_2", ptr @"\01L_OBJC_MODULES" ], section "llvm.metadata"		; <ptr> [#uses=0]

define internal ptr @"-[NSBitmapImageRep copyWithZone:]"(ptr %self, ptr %_cmd, ptr %zone) nounwind {
entry:
	%self_addr = alloca ptr		; <ptr> [#uses=2]
	%_cmd_addr = alloca ptr		; <ptr> [#uses=1]
	%zone_addr = alloca ptr		; <ptr> [#uses=2]
	%retval = alloca ptr		; <ptr> [#uses=1]
	%__block_holder_tmp_1.0 = alloca %struct.__block_1		; <ptr> [#uses=7]
	%new = alloca ptr		; <ptr> [#uses=2]
	%self.1 = alloca ptr		; <ptr> [#uses=2]
	%0 = alloca ptr		; <ptr> [#uses=2]
	%TRAMP.9 = alloca %struct.__builtin_trampoline, align 4		; <ptr> [#uses=1]
	%1 = alloca ptr		; <ptr> [#uses=2]
	%2 = alloca ptr		; <ptr> [#uses=2]
	%FRAME.7 = alloca %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]"		; <ptr> [#uses=5]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ptr %self, ptr %self_addr
	store ptr %_cmd, ptr %_cmd_addr
	store ptr %zone, ptr %zone_addr
	%3 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", ptr %FRAME.7, i32 0, i32 0		; <ptr> [#uses=1]
	%4 = load ptr, ptr %self_addr, align 4		; <ptr> [#uses=1]
	store ptr %4, ptr %3, align 4
	call void @llvm.init.trampoline(ptr %TRAMP.9, ptr @__helper_1.1632, ptr %FRAME.7)		; <ptr> [#uses=1]
        %tramp = call ptr @llvm.adjust.trampoline(ptr %TRAMP.9)
	store ptr %tramp, ptr %0, align 4
	%5 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", ptr %FRAME.7, i32 0, i32 1		; <ptr> [#uses=1]
	%6 = load ptr, ptr %0, align 4		; <ptr> [#uses=1]
	store ptr %6, ptr %5, align 4
	store ptr null, ptr %new, align 4
	%7 = getelementptr %struct.__block_1, ptr %__block_holder_tmp_1.0, i32 0, i32 0		; <ptr> [#uses=1]
	%8 = getelementptr %struct.__invoke_impl, ptr %7, i32 0, i32 0		; <ptr> [#uses=1]
	store ptr @_NSConcreteStackBlock, ptr %8, align 4
	%9 = getelementptr %struct.__block_1, ptr %__block_holder_tmp_1.0, i32 0, i32 0		; <ptr> [#uses=1]
	%10 = getelementptr %struct.__invoke_impl, ptr %9, i32 0, i32 1		; <ptr> [#uses=1]
	store i32 67108864, ptr %10, align 4
	%11 = getelementptr %struct.__block_1, ptr %__block_holder_tmp_1.0, i32 0, i32 0		; <ptr> [#uses=1]
	%12 = getelementptr %struct.__invoke_impl, ptr %11, i32 0, i32 2		; <ptr> [#uses=1]
	store i32 24, ptr %12, align 4
	%13 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", ptr %FRAME.7, i32 0, i32 1		; <ptr> [#uses=1]
	%14 = load ptr, ptr %13, align 4		; <ptr> [#uses=1]
	store ptr %14, ptr %1, align 4
	%15 = getelementptr %struct.__block_1, ptr %__block_holder_tmp_1.0, i32 0, i32 0		; <ptr> [#uses=1]
	%16 = getelementptr %struct.__invoke_impl, ptr %15, i32 0, i32 3		; <ptr> [#uses=1]
	%17 = load ptr, ptr %1, align 4		; <ptr> [#uses=1]
	store ptr %17, ptr %16, align 4
	%18 = getelementptr %struct.__block_1, ptr %__block_holder_tmp_1.0, i32 0, i32 1		; <ptr> [#uses=1]
	%19 = load ptr, ptr %zone_addr, align 4		; <ptr> [#uses=1]
	store ptr %19, ptr %18, align 4
	%20 = getelementptr %struct.__block_1, ptr %__block_holder_tmp_1.0, i32 0, i32 2		; <ptr> [#uses=1]
	store ptr %new, ptr %20, align 4
	%21 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", ptr %FRAME.7, i32 0, i32 0		; <ptr> [#uses=1]
	%22 = load ptr, ptr %21, align 4		; <ptr> [#uses=1]
	store ptr %22, ptr %2, align 4
	%23 = load ptr, ptr %2, align 4		; <ptr> [#uses=1]
	store ptr %23, ptr %self.1, align 4
	%24 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_1", align 4		; <ptr> [#uses=1]
	%25 = load ptr, ptr %self.1, align 4		; <ptr> [#uses=1]
	%26 = call ptr (ptr, ptr, ...) inttoptr (i64 4294901504 to ptr)(ptr %25, ptr %24, ptr %__block_holder_tmp_1.0) nounwind		; <ptr> [#uses=0]
	br label %return

return:		; preds = %entry
	%retval5 = load ptr, ptr %retval		; <ptr> [#uses=1]
	ret ptr %retval5
}

declare void @llvm.init.trampoline(ptr, ptr, ptr) nounwind
declare ptr @llvm.adjust.trampoline(ptr) nounwind

define internal void @__helper_1.1632(ptr nest %CHAIN.8, ptr %_self, ptr %cgImage) nounwind {
entry:
	%CHAIN.8_addr = alloca ptr		; <ptr> [#uses=2]
	%_self_addr = alloca ptr		; <ptr> [#uses=3]
	%cgImage_addr = alloca ptr		; <ptr> [#uses=1]
	%zone = alloca ptr		; <ptr> [#uses=2]
	%objc_super = alloca %struct._objc_super		; <ptr> [#uses=3]
	%new = alloca ptr		; <ptr> [#uses=2]
	%objc_super.5 = alloca ptr		; <ptr> [#uses=2]
	%0 = alloca ptr		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ptr %CHAIN.8, ptr %CHAIN.8_addr
	store ptr %_self, ptr %_self_addr
	store ptr %cgImage, ptr %cgImage_addr
	%1 = load ptr, ptr %_self_addr, align 4		; <ptr> [#uses=1]
	%2 = getelementptr %struct.__block_1, ptr %1, i32 0, i32 2		; <ptr> [#uses=1]
	%3 = load ptr, ptr %2, align 4		; <ptr> [#uses=1]
	store ptr %3, ptr %new, align 4
	%4 = load ptr, ptr %_self_addr, align 4		; <ptr> [#uses=1]
	%5 = getelementptr %struct.__block_1, ptr %4, i32 0, i32 1		; <ptr> [#uses=1]
	%6 = load ptr, ptr %5, align 4		; <ptr> [#uses=1]
	store ptr %6, ptr %zone, align 4
	%7 = load ptr, ptr %CHAIN.8_addr, align 4		; <ptr> [#uses=1]
	%8 = getelementptr %"struct.FRAME.-[NSBitmapImageRep copyWithZone:]", ptr %7, i32 0, i32 0		; <ptr> [#uses=1]
	%9 = load ptr, ptr %8, align 4		; <ptr> [#uses=1]
	store ptr %9, ptr %0, align 4
	%10 = load ptr, ptr %0, align 4		; <ptr> [#uses=1]
	%11 = getelementptr %struct._objc_super, ptr %objc_super, i32 0, i32 0		; <ptr> [#uses=1]
	store ptr %10, ptr %11, align 4
	%12 = load ptr, ptr getelementptr (%struct._objc_class, ptr @"\01L_OBJC_CLASS_NSBitmapImageRep", i32 0, i32 1), align 4		; <ptr> [#uses=1]
	%13 = getelementptr %struct._objc_super, ptr %objc_super, i32 0, i32 1		; <ptr> [#uses=1]
	store ptr %12, ptr %13, align 4
	store ptr %objc_super, ptr %objc_super.5, align 4
	%14 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_0", align 4		; <ptr> [#uses=1]
	%15 = load ptr, ptr %objc_super.5, align 4		; <ptr> [#uses=1]
	%16 = load ptr, ptr %zone, align 4		; <ptr> [#uses=1]
	%17 = call ptr (ptr, ptr, ...) @objc_msgSendSuper(ptr %15, ptr %14, ptr %16) nounwind		; <ptr> [#uses=1]
	%18 = load ptr, ptr %new, align 4		; <ptr> [#uses=1]
	store ptr %17, ptr %18, align 4
	br label %return

return:		; preds = %entry
	ret void
}

declare ptr @objc_msgSendSuper(ptr, ptr, ...)
