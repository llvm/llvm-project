; RUN: llvm-as < %s | llvm-dis -o /dev/null
	%0 = type { %object.ModuleInfo.__vtbl*, i8*, %"byte[]", %1, %"ClassInfo[]", i32, void ()*, void ()*, void ()*, i8*, void ()* }		; type %0
	%1 = type { i64, %object.ModuleInfo* }		; type %1
	%2 = type { i32, void ()* }		; type %2
	%"ClassInfo[]" = type { i64, %object.ClassInfo** }
	%"Interface[]" = type { i64, %object.Interface* }
	%"ModuleInfo[]" = type { i64, %object.ModuleInfo** }
	%ModuleReference = type { %ModuleReference*, %object.ModuleInfo* }
	%"OffsetTypeInfo[]" = type { i64, %object.OffsetTypeInfo* }
	%"byte[]" = type { i64, i8* }
	%object.ClassInfo = type { %object.ClassInfo.__vtbl*, i8*, %"byte[]", %"byte[]", %"void*[]", %"Interface[]", %object.ClassInfo*, i8*, i8*, i32, i8*, %"OffsetTypeInfo[]", i8*, %object.TypeInfo* }
	%object.ClassInfo.__vtbl = type { %object.ClassInfo*, %"byte[]" (%object.Object*)*, i64 (%object.Object*)*, i32 (%object.Object*, %object.Object*)*, i32 (%object.Object*, %object.Object*)*, %object.Object* (%object.ClassInfo*)* }
	%object.Interface = type { %object.ClassInfo*, %"void*[]", i64 }
	%object.ModuleInfo = type { %object.ModuleInfo.__vtbl*, i8*, %"byte[]", %"ModuleInfo[]", %"ClassInfo[]", i32, void ()*, void ()*, void ()*, i8*, void ()* }
	%object.ModuleInfo.__vtbl = type { %object.ClassInfo*, %"byte[]" (%object.Object*)*, i64 (%object.Object*)*, i32 (%object.Object*, %object.Object*)*, i32 (%object.Object*, %object.Object*)* }
	%object.Object = type { %object.ModuleInfo.__vtbl*, i8* }
	%object.OffsetTypeInfo = type { i64, %object.TypeInfo* }
	%object.TypeInfo = type { %object.TypeInfo.__vtbl*, i8* }
	%object.TypeInfo.__vtbl = type { %object.ClassInfo*, %"byte[]" (%object.Object*)*, i64 (%object.Object*)*, i32 (%object.Object*, %object.Object*)*, i32 (%object.Object*, %object.Object*)*, i64 (%object.TypeInfo*, i8*)*, i32 (%object.TypeInfo*, i8*, i8*)*, i32 (%object.TypeInfo*, i8*, i8*)*, i64 (%object.TypeInfo*)*, void (%object.TypeInfo*, i8*, i8*)*, %object.TypeInfo* (%object.TypeInfo*)*, %"byte[]" (%object.TypeInfo*)*, i32 (%object.TypeInfo*)*, %"OffsetTypeInfo[]" (%object.TypeInfo*)* }
	%"void*[]" = type { i64, i8** }
@_D10ModuleInfo6__vtblZ = external constant %object.ModuleInfo.__vtbl		; <%object.ModuleInfo.__vtbl*> [#uses=1]
@.str = internal constant [20 x i8] c"tango.core.BitManip\00"		; <[20 x i8]*> [#uses=1]
@_D5tango4core8BitManip8__ModuleZ = global %0 { %object.ModuleInfo.__vtbl* @_D10ModuleInfo6__vtblZ, i8* null, %"byte[]" { i64 19, i8* getelementptr ([20 x i8]* @.str, i32 0, i32 0) }, %1 zeroinitializer, %"ClassInfo[]" zeroinitializer, i32 4, void ()* null, void ()* null, void ()* null, i8* null, void ()* null }		; <%0*> [#uses=1]
@_D5tango4core8BitManip11__moduleRefZ = internal global %ModuleReference { %ModuleReference* null, %object.ModuleInfo* bitcast (%0* @_D5tango4core8BitManip8__ModuleZ to %object.ModuleInfo*) }		; <%ModuleReference*> [#uses=2]
@_Dmodule_ref = external global %ModuleReference*		; <%ModuleReference**> [#uses=2]
@llvm.global_ctors = appending constant [1 x %2] [%2 { i32 65535, void ()* @_D5tango4core8BitManip16__moduleinfoCtorZ }]		; <[1 x %2]*> [#uses=0]

define fastcc i32 @_D5tango4core8BitManip6popcntFkZi(i32 %x_arg) nounwind readnone {
entry:
	%tmp1 = lshr i32 %x_arg, 1		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 1431655765		; <i32> [#uses=1]
	%tmp4 = sub i32 %x_arg, %tmp2		; <i32> [#uses=2]
	%tmp6 = lshr i32 %tmp4, 2		; <i32> [#uses=1]
	%tmp7 = and i32 %tmp6, 858993459		; <i32> [#uses=1]
	%tmp9 = and i32 %tmp4, 858993459		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp7, %tmp9		; <i32> [#uses=2]
	%tmp12 = lshr i32 %tmp10, 4		; <i32> [#uses=1]
	%tmp14 = add i32 %tmp12, %tmp10		; <i32> [#uses=1]
	%tmp16 = and i32 %tmp14, 252645135		; <i32> [#uses=2]
	%tmp18 = lshr i32 %tmp16, 8		; <i32> [#uses=1]
	%tmp20 = add i32 %tmp18, %tmp16		; <i32> [#uses=1]
	%tmp22 = and i32 %tmp20, 16711935		; <i32> [#uses=2]
	%tmp24 = lshr i32 %tmp22, 16		; <i32> [#uses=1]
	%tmp26 = add i32 %tmp24, %tmp22		; <i32> [#uses=1]
	%tmp28 = and i32 %tmp26, 65535		; <i32> [#uses=1]
	ret i32 %tmp28
}

define fastcc i32 @_D5tango4core8BitManip7bitswapFkZk(i32 %x_arg) nounwind readnone {
entry:
	%tmp1 = lshr i32 %x_arg, 1		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 1431655765		; <i32> [#uses=1]
	%tmp4 = shl i32 %x_arg, 1		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, -1431655766		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=2]
	%tmp8 = lshr i32 %tmp6, 2		; <i32> [#uses=1]
	%tmp9 = and i32 %tmp8, 858993459		; <i32> [#uses=1]
	%tmp11 = shl i32 %tmp6, 2		; <i32> [#uses=1]
	%tmp12 = and i32 %tmp11, -858993460		; <i32> [#uses=1]
	%tmp13 = or i32 %tmp9, %tmp12		; <i32> [#uses=2]
	%tmp15 = lshr i32 %tmp13, 4		; <i32> [#uses=1]
	%tmp16 = and i32 %tmp15, 252645135		; <i32> [#uses=1]
	%tmp18 = shl i32 %tmp13, 4		; <i32> [#uses=1]
	%tmp19 = and i32 %tmp18, -252645136		; <i32> [#uses=1]
	%tmp20 = or i32 %tmp16, %tmp19		; <i32> [#uses=2]
	%tmp22 = lshr i32 %tmp20, 8		; <i32> [#uses=1]
	%tmp23 = and i32 %tmp22, 16711935		; <i32> [#uses=1]
	%tmp25 = shl i32 %tmp20, 8		; <i32> [#uses=1]
	%tmp26 = and i32 %tmp25, -16711936		; <i32> [#uses=1]
	%tmp27 = or i32 %tmp23, %tmp26		; <i32> [#uses=2]
	%tmp29 = lshr i32 %tmp27, 16		; <i32> [#uses=1]
	%tmp31 = shl i32 %tmp27, 16		; <i32> [#uses=1]
	%tmp32 = or i32 %tmp29, %tmp31		; <i32> [#uses=1]
	ret i32 %tmp32
}

define internal void @_D5tango4core8BitManip16__moduleinfoCtorZ() nounwind {
moduleinfoCtorEntry:
	%current = load %ModuleReference** @_Dmodule_ref		; <%ModuleReference*> [#uses=1]
	store %ModuleReference* %current, %ModuleReference** getelementptr (%ModuleReference* @_D5tango4core8BitManip11__moduleRefZ, i32 0, i32 0)
	store %ModuleReference* @_D5tango4core8BitManip11__moduleRefZ, %ModuleReference** @_Dmodule_ref
	ret void
}
!llvm.ldc.classinfo._D6Object7__ClassZ = !{!0}
!llvm.ldc.classinfo._D10ModuleInfo7__ClassZ = !{!1}
!0 = metadata !{%object.Object undef, i1 false, i1 false}
!1 = metadata !{%object.ModuleInfo undef, i1 false, i1 false}
