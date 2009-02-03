; RUN: llvm-as < %s | llc -march=thumb
; RUN: llvm-as < %s | llc -mtriple=thumb-apple-darwin -relocation-model=pic \
; RUN:   -mattr=+v6,+vfp2 | not grep {add r., r7, #2 \\* 4}

	%struct.__fooAllocator = type opaque
	%struct.__fooY = type { %struct.fooXBase, %struct.__fooString*, %struct.__fooU*, %struct.__fooV*, i8** }
	%struct.__fooZ = type opaque
	%struct.__fooU = type opaque
	%struct.__fooString = type opaque
	%struct.__fooV = type opaque
	%struct.fooXBase = type { i32, [4 x i8] }
	%struct.fooXClass = type { i32, i8*, void (i8*)*, i8* (%struct.__fooAllocator*, i8*)*, void (i8*)*, i8 (i8*, i8*) zeroext *, i32 (i8*)*, %struct.__fooString* (i8*, %struct.__fooZ*)*, %struct.__fooString* (i8*)* }
	%struct.aa_cache = type { i32, i32, [1 x %struct.aa_method*] }
	%struct.aa_class = type { %struct.aa_class*, %struct.aa_class*, i8*, i32, i32, i32, %struct.aa_ivar_list*, %struct.aa_method_list**, %struct.aa_cache*, %struct.aa_protocol_list* }
	%struct.aa_ivar = type { i8*, i8*, i32 }
	%struct.aa_ivar_list = type { i32, [1 x %struct.aa_ivar] }
	%struct.aa_method = type { %struct.aa_ss*, i8*, %struct.aa_object* (%struct.aa_object*, %struct.aa_ss*, ...)* }
	%struct.aa_method_list = type { %struct.aa_method_list*, i32, [1 x %struct.aa_method] }
	%struct.aa_object = type { %struct.aa_class* }
	%struct.aa_protocol_list = type { %struct.aa_protocol_list*, i32, [1 x %struct.aa_object*] }
	%struct.aa_ss = type opaque
@__kfooYTypeID = external global i32		; <i32*> [#uses=3]
@__fooYClass = external constant %struct.fooXClass		; <%struct.fooXClass*> [#uses=1]
@__fooXClassTableSize = external global i32		; <i32*> [#uses=1]
@__fooXAaClassTable = external global i32*		; <i32**> [#uses=1]
@s.10319 = external global %struct.aa_ss*		; <%struct.aa_ss**> [#uses=2]
@str15 = external constant [24 x i8]		; <[24 x i8]*> [#uses=1]


define i8 @test(%struct.__fooY* %calendar, double* %atp, i8* %componentDesc, ...) zeroext  {
entry:
	%args = alloca i8*, align 4		; <i8**> [#uses=5]
	%args4 = bitcast i8** %args to i8*		; <i8*> [#uses=2]
	call void @llvm.va_start( i8* %args4 )
	%tmp6 = load i32* @__kfooYTypeID		; <i32> [#uses=1]
	icmp eq i32 %tmp6, 0		; <i1>:0 [#uses=1]
	br i1 %0, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%tmp7 = call i32 @_fooXRegisterClass( %struct.fooXClass* @__fooYClass )		; <i32> [#uses=1]
	store i32 %tmp7, i32* @__kfooYTypeID
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp8 = load i32* @__kfooYTypeID		; <i32> [#uses=2]
	%tmp15 = load i32* @__fooXClassTableSize		; <i32> [#uses=1]
	icmp ugt i32 %tmp15, %tmp8		; <i1>:1 [#uses=1]
	br i1 %1, label %cond_next18, label %cond_true58

cond_next18:		; preds = %cond_next
	%tmp21 = getelementptr %struct.__fooY* %calendar, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp22 = load i32* %tmp21		; <i32> [#uses=2]
	%tmp29 = load i32** @__fooXAaClassTable		; <i32*> [#uses=1]
	%tmp31 = getelementptr i32* %tmp29, i32 %tmp8		; <i32*> [#uses=1]
	%tmp32 = load i32* %tmp31		; <i32> [#uses=1]
	icmp eq i32 %tmp22, %tmp32		; <i1>:2 [#uses=1]
	%.not = xor i1 %2, true		; <i1> [#uses=1]
	icmp ugt i32 %tmp22, 4095		; <i1>:3 [#uses=1]
	%bothcond = and i1 %.not, %3		; <i1> [#uses=1]
	br i1 %bothcond, label %cond_true58, label %bb48

bb48:		; preds = %cond_next18
	%tmp78 = call i32 @strlen( i8* %componentDesc )		; <i32> [#uses=4]
	%tmp92 = alloca i32, i32 %tmp78		; <i32*> [#uses=2]
	icmp sgt i32 %tmp78, 0		; <i1>:4 [#uses=1]
	br i1 %4, label %cond_true111, label %bb114

cond_true58:		; preds = %cond_next18, %cond_next
	%tmp59 = load %struct.aa_ss** @s.10319		; <%struct.aa_ss*> [#uses=2]
	icmp eq %struct.aa_ss* %tmp59, null		; <i1>:5 [#uses=1]
	%tmp6869 = bitcast %struct.__fooY* %calendar to i8*		; <i8*> [#uses=2]
	br i1 %5, label %cond_true60, label %cond_next64

cond_true60:		; preds = %cond_true58
	%tmp63 = call %struct.aa_ss* @sel_registerName( i8* getelementptr ([24 x i8]* @str15, i32 0, i32 0) )		; <%struct.aa_ss*> [#uses=2]
	store %struct.aa_ss* %tmp63, %struct.aa_ss** @s.10319
	%tmp66137 = volatile load i8** %args		; <i8*> [#uses=1]
	%tmp73138 = call i8 (i8*, %struct.aa_ss*, ...) zeroext * bitcast (%struct.aa_object* (%struct.aa_object*, %struct.aa_ss*, ...)* @aa_mm to i8 (i8*, %struct.aa_ss*, ...) zeroext *)( i8* %tmp6869, %struct.aa_ss* %tmp63, double* %atp, i8* %componentDesc, i8* %tmp66137) zeroext 		; <i8> [#uses=1]
	ret i8 %tmp73138

cond_next64:		; preds = %cond_true58
	%tmp66 = volatile load i8** %args		; <i8*> [#uses=1]
	%tmp73 = call i8 (i8*, %struct.aa_ss*, ...) zeroext * bitcast (%struct.aa_object* (%struct.aa_object*, %struct.aa_ss*, ...)* @aa_mm to i8 (i8*, %struct.aa_ss*, ...) zeroext *)( i8* %tmp6869, %struct.aa_ss* %tmp59, double* %atp, i8* %componentDesc, i8* %tmp66 ) zeroext 		; <i8> [#uses=1]
	ret i8 %tmp73

cond_true111:		; preds = %cond_true111, %bb48
	%idx.2132.0 = phi i32 [ 0, %bb48 ], [ %indvar.next, %cond_true111 ]		; <i32> [#uses=2]
	%tmp95 = volatile load i8** %args		; <i8*> [#uses=2]
	%tmp97 = getelementptr i8* %tmp95, i32 4		; <i8*> [#uses=1]
	volatile store i8* %tmp97, i8** %args
	%tmp9899 = bitcast i8* %tmp95 to i32*		; <i32*> [#uses=1]
	%tmp100 = load i32* %tmp9899		; <i32> [#uses=1]
	%tmp104 = getelementptr i32* %tmp92, i32 %idx.2132.0		; <i32*> [#uses=1]
	store i32 %tmp100, i32* %tmp104
	%indvar.next = add i32 %idx.2132.0, 1		; <i32> [#uses=2]
	icmp eq i32 %indvar.next, %tmp78		; <i1>:6 [#uses=1]
	br i1 %6, label %bb114, label %cond_true111

bb114:		; preds = %cond_true111, %bb48
	call void @llvm.va_end( i8* %args4 )
	%tmp122 = call i8 @_fooYCCV( %struct.__fooY* %calendar, double* %atp, i8* %componentDesc, i32* %tmp92, i32 %tmp78 ) zeroext 		; <i8> [#uses=1]
	ret i8 %tmp122
}

declare i32 @_fooXRegisterClass(%struct.fooXClass*)

declare i8 @_fooYCCV(%struct.__fooY*, double*, i8*, i32*, i32) zeroext 

declare %struct.aa_object* @aa_mm(%struct.aa_object*, %struct.aa_ss*, ...)

declare %struct.aa_ss* @sel_registerName(i8*)

declare void @llvm.va_start(i8*)

declare i32 @strlen(i8*)

declare void @llvm.va_end(i8*)
