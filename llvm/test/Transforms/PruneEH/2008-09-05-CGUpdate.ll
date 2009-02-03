; RUN: llvm-as < %s | opt -prune-eh -inline -print-callgraph \
; RUN:   -disable-output |& \
; RUN:     grep {Calls.*ce3806g__fxio__put__put_int64__4.1339} | count 2
	%struct.FRAME.ce3806g = type { %struct.string___XUB, %struct.string___XUB, %struct.string___XUB, %struct.string___XUB }
	%struct.FRAME.ce3806g__fxio__put__4 = type { i32, i32, i32, %struct.system__file_control_block__pstring*, i32, i32, i8 }
	%struct.RETURN = type { i8, i32 }
	%struct.ada__streams__root_stream_type = type { %struct.ada__tags__dispatch_table* }
	%struct.ada__tags__dispatch_table = type { [1 x i32] }
	%struct.ada__tags__select_specific_data = type { i32, %struct.ada__tags__select_specific_data_element }
	%struct.ada__tags__select_specific_data_element = type { i32, i8 }
	%struct.ada__tags__type_specific_data = type { i32, i32, [2147483647 x i8]*, [2147483647 x i8]*, %struct.ada__tags__dispatch_table*, i8, i32, i32, i32, i32, [2 x %struct.ada__tags__dispatch_table*] }
	%struct.ada__text_io__text_afcb = type { %struct.system__file_control_block__afcb, i32, i32, i32, i32, i32, %struct.ada__text_io__text_afcb*, i8, i8 }
	%struct.exception = type { i8, i8, i32, i8*, i8*, i32, i8* }
	%struct.long_long_float___PAD = type { x86_fp80, [1 x i32] }
	%struct.string___XUB = type { i32, i32 }
	%struct.system__file_control_block__afcb = type { %struct.ada__streams__root_stream_type, i32, %struct.system__file_control_block__pstring, %struct.system__file_control_block__pstring, i8, i8, i8, i8, i8, i8, i8, %struct.system__file_control_block__afcb*, %struct.system__file_control_block__afcb* }
	%struct.system__file_control_block__pstring = type { i8*, %struct.string___XUB* }
	%struct.system__finalization_implementation__limited_record_controller = type { %struct.system__finalization_root__root_controlled, %struct.system__finalization_root__root_controlled* }
	%struct.system__finalization_implementation__record_controller = type { %struct.system__finalization_implementation__limited_record_controller, i32 }
	%struct.system__finalization_root__empty_root_controlled = type { %struct.ada__tags__dispatch_table* }
	%struct.system__finalization_root__root_controlled = type { %struct.ada__streams__root_stream_type, %struct.system__finalization_root__root_controlled*, %struct.system__finalization_root__root_controlled* }
	%struct.system__secondary_stack__mark_id = type { i32, i32 }
	%struct.system__standard_library__exception_data = type { i8, i8, i32, i32, %struct.system__standard_library__exception_data*, i32, void ()* }
@.str = internal constant [12 x i8] c"system.ads\00\00"		; <[12 x i8]*> [#uses=1]
@.str1 = internal constant [14 x i8] c"a-tifiio.adb\00\00"		; <[14 x i8]*> [#uses=1]
@system__soft_links__abort_undefer = external global void ()*		; <void ()**> [#uses=6]
@.str2 = internal constant [47 x i8] c"a-tifiio.adb:327 instantiated at ce3806g.adb:52"		; <[47 x i8]*> [#uses=1]
@C.354.2200 = internal constant %struct.string___XUB { i32 1, i32 47 }		; <%struct.string___XUB*> [#uses=1]
@ada__io_exceptions__data_error = external global %struct.exception		; <%struct.exception*> [#uses=1]
@constraint_error = external global %struct.exception		; <%struct.exception*> [#uses=2]
@__gnat_all_others_value = external constant i32		; <i32*> [#uses=21]
@.str3 = internal constant [10 x i8] c"0123456789"		; <[10 x i8]*> [#uses=2]
@ada__text_io__current_out = external global %struct.ada__text_io__text_afcb*		; <%struct.ada__text_io__text_afcb**> [#uses=1]
@.str4 = internal constant [126 x i8] c"CHECK THAT FIXED_IO PUT OPERATES ON FILES OF MODE OUT_FILE AND IF NO FILE IS SPECIFIED THE CURRENT DEFAULT OUTPUT FILE IS USED"		; <[126 x i8]*> [#uses=1]
@C.131.1559 = internal constant %struct.string___XUB { i32 1, i32 126 }		; <%struct.string___XUB*> [#uses=1]
@.str5 = internal constant [7 x i8] c"CE3806G"		; <[7 x i8]*> [#uses=1]
@C.132.1562 = internal constant %struct.string___XUB { i32 1, i32 7 }		; <%struct.string___XUB*> [#uses=1]
@incompleteF.1176.b = internal global i1 false		; <i1*> [#uses=2]
@incomplete.1177 = internal global %struct.exception { i8 0, i8 65, i32 23, i8* getelementptr ([23 x i8]* @incompleteE.1174, i32 0, i32 0), i8* null, i32 0, i8* null }		; <%struct.exception*> [#uses=15]
@incompleteE.1174 = internal global [23 x i8] c"CE3806G.B_1.INCOMPLETE\00"		; <[23 x i8]*> [#uses=1]
@.str6 = internal constant [0 x i8] zeroinitializer		; <[0 x i8]*> [#uses=1]
@C.136.1568 = internal constant %struct.string___XUB { i32 1, i32 0 }		; <%struct.string___XUB*> [#uses=1]
@C.137.1571 = internal constant %struct.string___XUB { i32 1, i32 0 }		; <%struct.string___XUB*> [#uses=1]
@.str7 = internal constant [50 x i8] c"USE_ERROR RAISED ON TEXT CREATE WITH OUT_FILE MODE"		; <[50 x i8]*> [#uses=1]
@C.139.1577 = internal constant %struct.string___XUB { i32 1, i32 50 }		; <%struct.string___XUB*> [#uses=1]
@.str8 = internal constant [14 x i8] c"ce3806g.adb:65"		; <[14 x i8]*> [#uses=1]
@C.140.1580 = internal constant %struct.string___XUB { i32 1, i32 14 }		; <%struct.string___XUB*> [#uses=1]
@.str9 = internal constant [51 x i8] c"NAME_ERROR RAISED ON TEXT CREATE WITH OUT_FILE MODE"		; <[51 x i8]*> [#uses=1]
@C.143.1585 = internal constant %struct.string___XUB { i32 1, i32 51 }		; <%struct.string___XUB*> [#uses=1]
@.str10 = internal constant [14 x i8] c"ce3806g.adb:69"		; <[14 x i8]*> [#uses=1]
@C.144.1588 = internal constant %struct.string___XUB { i32 1, i32 14 }		; <%struct.string___XUB*> [#uses=1]
@C.146.1592 = internal constant %struct.string___XUB { i32 1, i32 0 }		; <%struct.string___XUB*> [#uses=1]
@C.147.1595 = internal constant %struct.string___XUB { i32 1, i32 0 }		; <%struct.string___XUB*> [#uses=1]
@C.153.1609 = internal constant %struct.string___XUB { i32 1, i32 0 }		; <%struct.string___XUB*> [#uses=1]
@C.154.1612 = internal constant %struct.string___XUB { i32 1, i32 0 }		; <%struct.string___XUB*> [#uses=1]
@.str12 = internal constant [47 x i8] c"USE_ERROR RAISED ON TEXT OPEN WITH IN_FILE MODE"		; <[47 x i8]*> [#uses=1]
@C.156.1618 = internal constant %struct.string___XUB { i32 1, i32 47 }		; <%struct.string___XUB*> [#uses=1]
@.str13 = internal constant [14 x i8] c"ce3806g.adb:88"		; <[14 x i8]*> [#uses=1]
@C.157.1621 = internal constant %struct.string___XUB { i32 1, i32 14 }		; <%struct.string___XUB*> [#uses=1]
@C.159.1627 = internal constant %struct.string___XUB { i32 1, i32 0 }		; <%struct.string___XUB*> [#uses=1]
@C.160.1630 = internal constant %struct.string___XUB { i32 1, i32 0 }		; <%struct.string___XUB*> [#uses=1]
@.str14 = internal constant [33 x i8] c"VALUE INCORRECT - FIXED FROM FILE"		; <[33 x i8]*> [#uses=1]
@C.162.1637 = internal constant %struct.string___XUB { i32 1, i32 33 }		; <%struct.string___XUB*> [#uses=1]
@.str15 = internal constant [36 x i8] c"VALUE INCORRECT - FIXED FROM DEFAULT"		; <[36 x i8]*> [#uses=1]
@C.164.1642 = internal constant %struct.string___XUB { i32 1, i32 36 }		; <%struct.string___XUB*> [#uses=1]
@ada__io_exceptions__use_error = external global %struct.exception		; <%struct.exception*> [#uses=4]
@ada__io_exceptions__name_error = external global %struct.exception		; <%struct.exception*> [#uses=2]

define void @_ada_ce3806g() {
entry:
	%0 = alloca %struct.system__file_control_block__pstring, align 8		; <%struct.system__file_control_block__pstring*> [#uses=3]
	%1 = alloca %struct.system__file_control_block__pstring, align 8		; <%struct.system__file_control_block__pstring*> [#uses=3]
	%2 = alloca %struct.system__file_control_block__pstring, align 8		; <%struct.system__file_control_block__pstring*> [#uses=3]
	%3 = alloca %struct.system__file_control_block__pstring, align 8		; <%struct.system__file_control_block__pstring*> [#uses=3]
	%FRAME.356 = alloca %struct.FRAME.ce3806g		; <%struct.FRAME.ce3806g*> [#uses=20]
	call void @report__test( i8* getelementptr ([7 x i8]* @.str5, i32 0, i32 0), %struct.string___XUB* @C.132.1562, i8* getelementptr ([126 x i8]* @.str4, i32 0, i32 0), %struct.string___XUB* @C.131.1559 )
	%4 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3		; <%struct.string___XUB*> [#uses=1]
	call void @system__secondary_stack__ss_mark( %struct.string___XUB* noalias sret %4 )
	%.b = load i1* @incompleteF.1176.b		; <i1> [#uses=1]
	br i1 %.b, label %bb11, label %bb

bb:		; preds = %entry
	invoke void @system__exception_table__register_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @incomplete.1177 to %struct.system__standard_library__exception_data*) )
			to label %invcont unwind label %lpad

invcont:		; preds = %bb
	store i1 true, i1* @incompleteF.1176.b
	br label %bb11

bb11:		; preds = %entry, %invcont
	%5 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2		; <%struct.string___XUB*> [#uses=1]
	invoke void @system__secondary_stack__ss_mark( %struct.string___XUB* noalias sret %5 )
			to label %invcont12 unwind label %lpad228

invcont12:		; preds = %bb11
	invoke void @report__legal_file_name( %struct.system__file_control_block__pstring* noalias sret %3, i32 1, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.137.1571 )
			to label %invcont17 unwind label %lpad232

invcont17:		; preds = %invcont12
	%elt18 = getelementptr %struct.system__file_control_block__pstring* %3, i32 0, i32 0		; <i8**> [#uses=1]
	%val19 = load i8** %elt18, align 8		; <i8*> [#uses=1]
	%elt20 = getelementptr %struct.system__file_control_block__pstring* %3, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	%val21 = load %struct.string___XUB** %elt20		; <%struct.string___XUB*> [#uses=1]
	%6 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__create( %struct.ada__text_io__text_afcb* null, i8 2, i8* %val19, %struct.string___XUB* %val21, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.136.1568 )
			to label %invcont26 unwind label %lpad232		; <%struct.ada__text_io__text_afcb*> [#uses=2]

invcont26:		; preds = %invcont17
	%7 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2, i32 0		; <i32*> [#uses=1]
	%8 = load i32* %7, align 8		; <i32> [#uses=1]
	%9 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2, i32 1		; <i32*> [#uses=1]
	%10 = load i32* %9, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %8, i32 %10 )
			to label %bb73 unwind label %lpad228

bb32:		; preds = %lpad232
	call void @__gnat_begin_handler( i8* %eh_ptr233 ) nounwind
	%11 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %11( )
			to label %invcont33 unwind label %lpad240

invcont33:		; preds = %bb32
	invoke void @report__not_applicable( i8* getelementptr ([50 x i8]* @.str7, i32 0, i32 0), %struct.string___XUB* @C.139.1577 )
			to label %invcont38 unwind label %lpad240

invcont38:		; preds = %invcont33
	invoke void @__gnat_raise_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @incomplete.1177 to %struct.system__standard_library__exception_data*), i8* getelementptr ([14 x i8]* @.str8, i32 0, i32 0), %struct.string___XUB* @C.140.1580 ) noreturn
			to label %invcont43 unwind label %lpad240

invcont43:		; preds = %invcont38
	unreachable

bb47:		; preds = %ppad291
	call void @__gnat_begin_handler( i8* %eh_ptr233 ) nounwind
	%12 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %12( )
			to label %invcont49 unwind label %lpad248

invcont49:		; preds = %bb47
	invoke void @report__not_applicable( i8* getelementptr ([51 x i8]* @.str9, i32 0, i32 0), %struct.string___XUB* @C.143.1585 )
			to label %invcont54 unwind label %lpad248

invcont54:		; preds = %invcont49
	invoke void @__gnat_raise_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @incomplete.1177 to %struct.system__standard_library__exception_data*), i8* getelementptr ([14 x i8]* @.str10, i32 0, i32 0), %struct.string___XUB* @C.144.1588 ) noreturn
			to label %invcont59 unwind label %lpad248

invcont59:		; preds = %invcont54
	unreachable

bb73:		; preds = %invcont26
	invoke void @report__legal_file_name( %struct.system__file_control_block__pstring* noalias sret %2, i32 2, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.147.1595 )
			to label %invcont78 unwind label %lpad228

invcont78:		; preds = %bb73
	%elt79 = getelementptr %struct.system__file_control_block__pstring* %2, i32 0, i32 0		; <i8**> [#uses=1]
	%val80 = load i8** %elt79, align 8		; <i8*> [#uses=1]
	%elt81 = getelementptr %struct.system__file_control_block__pstring* %2, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	%val82 = load %struct.string___XUB** %elt81		; <%struct.string___XUB*> [#uses=1]
	%13 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__create( %struct.ada__text_io__text_afcb* null, i8 2, i8* %val80, %struct.string___XUB* %val82, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.146.1592 )
			to label %invcont87 unwind label %lpad228		; <%struct.ada__text_io__text_afcb*> [#uses=2]

invcont87:		; preds = %invcont78
	invoke void @ada__text_io__set_output( %struct.ada__text_io__text_afcb* %13 )
			to label %invcont88 unwind label %lpad228

invcont88:		; preds = %invcont87
	%14 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1		; <%struct.string___XUB*> [#uses=1]
	invoke void @system__secondary_stack__ss_mark( %struct.string___XUB* noalias sret %14 )
			to label %invcont89 unwind label %lpad228

invcont89:		; preds = %invcont88
	invoke fastcc void @ce3806g__fxio__put.1149( %struct.ada__text_io__text_afcb* %6 )
			to label %bb94 unwind label %lpad252

bb94:		; preds = %invcont89
	invoke fastcc void @ce3806g__fxio__put__2.1155( )
			to label %invcont95 unwind label %lpad252

invcont95:		; preds = %bb94
	%15 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__close( %struct.ada__text_io__text_afcb* %6 )
			to label %invcont96 unwind label %lpad252		; <%struct.ada__text_io__text_afcb*> [#uses=1]

invcont96:		; preds = %invcont95
	%16 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0		; <%struct.string___XUB*> [#uses=1]
	invoke void @system__secondary_stack__ss_mark( %struct.string___XUB* noalias sret %16 )
			to label %invcont97 unwind label %lpad252

invcont97:		; preds = %invcont96
	invoke void @report__legal_file_name( %struct.system__file_control_block__pstring* noalias sret %1, i32 1, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.154.1612 )
			to label %invcont102 unwind label %lpad256

invcont102:		; preds = %invcont97
	%elt103 = getelementptr %struct.system__file_control_block__pstring* %1, i32 0, i32 0		; <i8**> [#uses=1]
	%val104 = load i8** %elt103, align 8		; <i8*> [#uses=1]
	%elt105 = getelementptr %struct.system__file_control_block__pstring* %1, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	%val106 = load %struct.string___XUB** %elt105		; <%struct.string___XUB*> [#uses=1]
	%17 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__open( %struct.ada__text_io__text_afcb* %15, i8 0, i8* %val104, %struct.string___XUB* %val106, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.153.1609 )
			to label %invcont111 unwind label %lpad256		; <%struct.ada__text_io__text_afcb*> [#uses=2]

invcont111:		; preds = %invcont102
	%18 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%19 = load i32* %18, align 8		; <i32> [#uses=1]
	%20 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%21 = load i32* %20, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %19, i32 %21 )
			to label %bb143 unwind label %lpad252

bb117:		; preds = %lpad256
	call void @__gnat_begin_handler( i8* %eh_ptr257 ) nounwind
	%22 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %22( )
			to label %invcont119 unwind label %lpad264

invcont119:		; preds = %bb117
	invoke void @report__not_applicable( i8* getelementptr ([47 x i8]* @.str12, i32 0, i32 0), %struct.string___XUB* @C.156.1618 )
			to label %invcont124 unwind label %lpad264

invcont124:		; preds = %invcont119
	invoke void @__gnat_raise_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @incomplete.1177 to %struct.system__standard_library__exception_data*), i8* getelementptr ([14 x i8]* @.str13, i32 0, i32 0), %struct.string___XUB* @C.157.1621 ) noreturn
			to label %invcont129 unwind label %lpad264

invcont129:		; preds = %invcont124
	unreachable

bb143:		; preds = %invcont111
	%23 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__standard_output( )
			to label %invcont144 unwind label %lpad252		; <%struct.ada__text_io__text_afcb*> [#uses=1]

invcont144:		; preds = %bb143
	invoke void @ada__text_io__set_output( %struct.ada__text_io__text_afcb* %23 )
			to label %invcont145 unwind label %lpad252

invcont145:		; preds = %invcont144
	%24 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__close( %struct.ada__text_io__text_afcb* %13 )
			to label %invcont146 unwind label %lpad252		; <%struct.ada__text_io__text_afcb*> [#uses=1]

invcont146:		; preds = %invcont145
	invoke void @report__legal_file_name( %struct.system__file_control_block__pstring* noalias sret %0, i32 2, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.160.1630 )
			to label %invcont151 unwind label %lpad252

invcont151:		; preds = %invcont146
	%elt152 = getelementptr %struct.system__file_control_block__pstring* %0, i32 0, i32 0		; <i8**> [#uses=1]
	%val153 = load i8** %elt152, align 8		; <i8*> [#uses=1]
	%elt154 = getelementptr %struct.system__file_control_block__pstring* %0, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	%val155 = load %struct.string___XUB** %elt154		; <%struct.string___XUB*> [#uses=1]
	%25 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__open( %struct.ada__text_io__text_afcb* %24, i8 0, i8* %val153, %struct.string___XUB* %val155, i8* getelementptr ([0 x i8]* @.str6, i32 0, i32 0), %struct.string___XUB* @C.159.1627 )
			to label %invcont160 unwind label %lpad252		; <%struct.ada__text_io__text_afcb*> [#uses=2]

invcont160:		; preds = %invcont151
	%26 = invoke fastcc i8 @ce3806g__fxio__get.1137( %struct.ada__text_io__text_afcb* %17 ) signext
			to label %invcont161 unwind label %lpad252		; <i8> [#uses=1]

invcont161:		; preds = %invcont160
	%27 = icmp eq i8 %26, -3		; <i1> [#uses=1]
	br i1 %27, label %bb169, label %bb163

bb163:		; preds = %invcont161
	invoke void @report__failed( i8* getelementptr ([33 x i8]* @.str14, i32 0, i32 0), %struct.string___XUB* @C.162.1637 )
			to label %bb169 unwind label %lpad252

bb169:		; preds = %invcont161, %bb163
	%28 = invoke fastcc i8 @ce3806g__fxio__get.1137( %struct.ada__text_io__text_afcb* %25 ) signext
			to label %invcont170 unwind label %lpad252		; <i8> [#uses=1]

invcont170:		; preds = %bb169
	%29 = icmp eq i8 %28, -1		; <i1> [#uses=1]
	br i1 %29, label %bb187, label %bb172

bb172:		; preds = %invcont170
	invoke void @report__failed( i8* getelementptr ([36 x i8]* @.str15, i32 0, i32 0), %struct.string___XUB* @C.164.1642 )
			to label %bb187 unwind label %lpad252

bb187:		; preds = %invcont170, %bb172
	%30 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1, i32 0		; <i32*> [#uses=1]
	%31 = load i32* %30, align 8		; <i32> [#uses=1]
	%32 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1, i32 1		; <i32*> [#uses=1]
	%33 = load i32* %32, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %31, i32 %33 )
			to label %bb193 unwind label %lpad228

bb193:		; preds = %bb187
	%34 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__delete( %struct.ada__text_io__text_afcb* %17 )
			to label %invcont194 unwind label %lpad268		; <%struct.ada__text_io__text_afcb*> [#uses=0]

invcont194:		; preds = %bb193
	%35 = invoke %struct.ada__text_io__text_afcb* @ada__text_io__delete( %struct.ada__text_io__text_afcb* %25 )
			to label %bb221 unwind label %lpad268		; <%struct.ada__text_io__text_afcb*> [#uses=0]

bb196:		; preds = %lpad268
	call void @__gnat_begin_handler( i8* %eh_ptr269 ) nounwind
	%36 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %36( )
			to label %bb203 unwind label %lpad276

bb203:		; preds = %bb196
	invoke void @__gnat_end_handler( i8* %eh_ptr269 )
			to label %bb221 unwind label %lpad272

bb205:		; preds = %ppad304
	call void @__gnat_begin_handler( i8* %eh_exception.1 ) nounwind
	%37 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %37( )
			to label %bb212 unwind label %lpad284

bb212:		; preds = %bb205
	invoke void @__gnat_end_handler( i8* %eh_exception.1 )
			to label %bb221 unwind label %lpad280

bb221:		; preds = %invcont194, %bb212, %bb203
	%38 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3, i32 0		; <i32*> [#uses=1]
	%39 = load i32* %38, align 8		; <i32> [#uses=1]
	%40 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3, i32 1		; <i32*> [#uses=1]
	%41 = load i32* %40, align 4		; <i32> [#uses=1]
	call void @system__secondary_stack__ss_release( i32 %39, i32 %41 )
	call void @report__result( )
	ret void

lpad:		; preds = %bb
	%eh_ptr = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select227 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	br label %ppad

lpad228:		; preds = %bb187, %ppad294, %invcont88, %invcont87, %invcont78, %bb73, %ppad288, %invcont26, %bb11
	%eh_ptr229 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select231 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr229, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad304

lpad232:		; preds = %invcont17, %invcont12
	%eh_ptr233 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=6]
	%eh_select235 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr233, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @ada__io_exceptions__use_error, %struct.exception* @ada__io_exceptions__name_error, %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=3]
	%eh_typeid = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @ada__io_exceptions__use_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%42 = icmp eq i32 %eh_select235, %eh_typeid		; <i1> [#uses=1]
	br i1 %42, label %bb32, label %ppad291

lpad236:		; preds = %lpad240
	%eh_ptr237 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select239 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr237, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad288

lpad240:		; preds = %invcont38, %invcont33, %bb32
	%eh_ptr241 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select243 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr241, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	invoke void @__gnat_end_handler( i8* %eh_ptr233 )
			to label %ppad288 unwind label %lpad236

lpad244:		; preds = %lpad248
	%eh_ptr245 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select247 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr245, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad288

lpad248:		; preds = %invcont54, %invcont49, %bb47
	%eh_ptr249 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select251 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr249, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	invoke void @__gnat_end_handler( i8* %eh_ptr233 )
			to label %ppad288 unwind label %lpad244

lpad252:		; preds = %bb94, %invcont89, %invcont160, %bb169, %bb172, %bb163, %invcont151, %invcont146, %invcont145, %invcont144, %bb143, %ppad295, %invcont111, %invcont96, %invcont95
	%eh_ptr253 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select255 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr253, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad294

lpad256:		; preds = %invcont102, %invcont97
	%eh_ptr257 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=4]
	%eh_select259 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr257, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @ada__io_exceptions__use_error, %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=2]
	%eh_typeid297 = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @ada__io_exceptions__use_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%43 = icmp eq i32 %eh_select259, %eh_typeid297		; <i1> [#uses=1]
	br i1 %43, label %bb117, label %ppad295

lpad260:		; preds = %lpad264
	%eh_ptr261 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select263 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr261, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad295

lpad264:		; preds = %invcont124, %invcont119, %bb117
	%eh_ptr265 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select267 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr265, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	invoke void @__gnat_end_handler( i8* %eh_ptr257 )
			to label %ppad295 unwind label %lpad260

lpad268:		; preds = %invcont194, %bb193
	%eh_ptr269 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=5]
	%eh_select271 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr269, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @ada__io_exceptions__use_error, %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=2]
	%eh_typeid301 = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @ada__io_exceptions__use_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%44 = icmp eq i32 %eh_select271, %eh_typeid301		; <i1> [#uses=1]
	br i1 %44, label %bb196, label %ppad304

lpad272:		; preds = %bb203, %lpad276
	%eh_ptr273 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select275 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr273, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	br label %ppad304

lpad276:		; preds = %bb196
	%eh_ptr277 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select279 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr277, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @incomplete.1177, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	invoke void @__gnat_end_handler( i8* %eh_ptr269 )
			to label %ppad304 unwind label %lpad272

lpad280:		; preds = %bb212, %lpad284
	%eh_ptr281 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select283 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr281, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	br label %ppad

lpad284:		; preds = %bb205
	%eh_ptr285 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select287 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr285, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	invoke void @__gnat_end_handler( i8* %eh_exception.1 )
			to label %ppad unwind label %lpad280

ppad:		; preds = %lpad284, %ppad304, %lpad280, %lpad
	%eh_exception.2 = phi i8* [ %eh_exception.1, %ppad304 ], [ %eh_ptr281, %lpad280 ], [ %eh_ptr, %lpad ], [ %eh_ptr285, %lpad284 ]		; <i8*> [#uses=1]
	%45 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3, i32 0		; <i32*> [#uses=1]
	%46 = load i32* %45, align 8		; <i32> [#uses=1]
	%47 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 3, i32 1		; <i32*> [#uses=1]
	%48 = load i32* %47, align 4		; <i32> [#uses=1]
	call void @system__secondary_stack__ss_release( i32 %46, i32 %48 )
	%49 = call i32 (...)* @_Unwind_Resume( i8* %eh_exception.2 )		; <i32> [#uses=0]
	unreachable

ppad288:		; preds = %lpad248, %lpad240, %ppad291, %lpad244, %lpad236
	%eh_exception.0 = phi i8* [ %eh_ptr233, %ppad291 ], [ %eh_ptr245, %lpad244 ], [ %eh_ptr237, %lpad236 ], [ %eh_ptr241, %lpad240 ], [ %eh_ptr249, %lpad248 ]		; <i8*> [#uses=1]
	%eh_selector.0 = phi i32 [ %eh_select235, %ppad291 ], [ %eh_select247, %lpad244 ], [ %eh_select239, %lpad236 ], [ %eh_select243, %lpad240 ], [ %eh_select251, %lpad248 ]		; <i32> [#uses=1]
	%50 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2, i32 0		; <i32*> [#uses=1]
	%51 = load i32* %50, align 8		; <i32> [#uses=1]
	%52 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 2, i32 1		; <i32*> [#uses=1]
	%53 = load i32* %52, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %51, i32 %53 )
			to label %ppad304 unwind label %lpad228

ppad291:		; preds = %lpad232
	%eh_typeid292 = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @ada__io_exceptions__name_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%54 = icmp eq i32 %eh_select235, %eh_typeid292		; <i1> [#uses=1]
	br i1 %54, label %bb47, label %ppad288

ppad294:		; preds = %ppad295, %lpad252
	%eh_exception.4 = phi i8* [ %eh_ptr253, %lpad252 ], [ %eh_exception.3, %ppad295 ]		; <i8*> [#uses=1]
	%eh_selector.4 = phi i32 [ %eh_select255, %lpad252 ], [ %eh_selector.3, %ppad295 ]		; <i32> [#uses=1]
	%55 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1, i32 0		; <i32*> [#uses=1]
	%56 = load i32* %55, align 8		; <i32> [#uses=1]
	%57 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 1, i32 1		; <i32*> [#uses=1]
	%58 = load i32* %57, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %56, i32 %58 )
			to label %ppad304 unwind label %lpad228

ppad295:		; preds = %lpad264, %lpad256, %lpad260
	%eh_exception.3 = phi i8* [ %eh_ptr257, %lpad256 ], [ %eh_ptr261, %lpad260 ], [ %eh_ptr265, %lpad264 ]		; <i8*> [#uses=1]
	%eh_selector.3 = phi i32 [ %eh_select259, %lpad256 ], [ %eh_select263, %lpad260 ], [ %eh_select267, %lpad264 ]		; <i32> [#uses=1]
	%59 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0, i32 0		; <i32*> [#uses=1]
	%60 = load i32* %59, align 8		; <i32> [#uses=1]
	%61 = getelementptr %struct.FRAME.ce3806g* %FRAME.356, i32 0, i32 0, i32 1		; <i32*> [#uses=1]
	%62 = load i32* %61, align 4		; <i32> [#uses=1]
	invoke void @system__secondary_stack__ss_release( i32 %60, i32 %62 )
			to label %ppad294 unwind label %lpad252

ppad304:		; preds = %lpad276, %ppad294, %ppad288, %lpad268, %lpad272, %lpad228
	%eh_exception.1 = phi i8* [ %eh_ptr229, %lpad228 ], [ %eh_ptr269, %lpad268 ], [ %eh_ptr273, %lpad272 ], [ %eh_exception.0, %ppad288 ], [ %eh_exception.4, %ppad294 ], [ %eh_ptr277, %lpad276 ]		; <i8*> [#uses=4]
	%eh_selector.1 = phi i32 [ %eh_select231, %lpad228 ], [ %eh_select271, %lpad268 ], [ %eh_select275, %lpad272 ], [ %eh_selector.0, %ppad288 ], [ %eh_selector.4, %ppad294 ], [ %eh_select279, %lpad276 ]		; <i32> [#uses=1]
	%eh_typeid305 = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @incomplete.1177, i32 0, i32 0) )		; <i32> [#uses=1]
	%63 = icmp eq i32 %eh_selector.1, %eh_typeid305		; <i1> [#uses=1]
	br i1 %63, label %bb205, label %ppad
}

define internal fastcc i8 @ce3806g__fxio__get.1137(%struct.ada__text_io__text_afcb* %file) signext {
entry:
	%0 = invoke x86_fp80 @ada__text_io__float_aux__get( %struct.ada__text_io__text_afcb* %file, i32 0 )
			to label %invcont unwind label %lpad		; <x86_fp80> [#uses=5]

invcont:		; preds = %entry
	%1 = fcmp ult x86_fp80 %0, 0xKFFFEFFFFFFFFFFFFFFFF		; <i1> [#uses=1]
	%2 = fcmp ugt x86_fp80 %0, 0xK7FFEFFFFFFFFFFFFFFFF		; <i1> [#uses=1]
	%or.cond = or i1 %1, %2		; <i1> [#uses=1]
	br i1 %or.cond, label %bb2, label %bb4

bb2:		; preds = %invcont
	invoke void @__gnat_rcheck_12( i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0), i32 1 ) noreturn
			to label %invcont3 unwind label %lpad

invcont3:		; preds = %bb2
	unreachable

bb4:		; preds = %invcont
	%3 = mul x86_fp80 %0, 0xK40008000000000000000		; <x86_fp80> [#uses=1]
	%4 = fcmp ult x86_fp80 %3, 0xKC0068000000000000000		; <i1> [#uses=1]
	br i1 %4, label %bb8, label %bb6

bb6:		; preds = %bb4
	%5 = mul x86_fp80 %0, 0xK40008000000000000000		; <x86_fp80> [#uses=1]
	%6 = fcmp ugt x86_fp80 %5, 0xK4005FE00000000000000		; <i1> [#uses=1]
	br i1 %6, label %bb8, label %bb10

bb8:		; preds = %bb4, %bb6
	invoke void @__gnat_rcheck_10( i8* getelementptr ([14 x i8]* @.str1, i32 0, i32 0), i32 324 ) noreturn
			to label %invcont9 unwind label %lpad

invcont9:		; preds = %bb8
	unreachable

bb10:		; preds = %bb6
	%7 = mul x86_fp80 %0, 0xK40008000000000000000		; <x86_fp80> [#uses=3]
	%8 = fcmp ult x86_fp80 %7, 0xK00000000000000000000		; <i1> [#uses=1]
	br i1 %8, label %bb13, label %bb12

bb12:		; preds = %bb10
	%9 = add x86_fp80 %7, 0xK3FFDFFFFFFFFFFFFFFFF		; <x86_fp80> [#uses=1]
	br label %bb14

bb13:		; preds = %bb10
	%10 = sub x86_fp80 %7, 0xK3FFDFFFFFFFFFFFFFFFF		; <x86_fp80> [#uses=1]
	br label %bb14

bb14:		; preds = %bb13, %bb12
	%iftmp.339.0.in = phi x86_fp80 [ %10, %bb13 ], [ %9, %bb12 ]		; <x86_fp80> [#uses=1]
	%iftmp.339.0 = fptosi x86_fp80 %iftmp.339.0.in to i8		; <i8> [#uses=3]
	%11 = add i8 %iftmp.339.0, 20		; <i8> [#uses=1]
	%12 = icmp ugt i8 %11, 40		; <i1> [#uses=1]
	br i1 %12, label %bb16, label %bb18

bb16:		; preds = %bb14
	invoke void @__gnat_rcheck_12( i8* getelementptr ([14 x i8]* @.str1, i32 0, i32 0), i32 324 ) noreturn
			to label %invcont17 unwind label %lpad

invcont17:		; preds = %bb16
	unreachable

bb18:		; preds = %bb14
	%13 = add i8 %iftmp.339.0, 20		; <i8> [#uses=1]
	%14 = icmp ugt i8 %13, 40		; <i1> [#uses=1]
	br i1 %14, label %bb20, label %bb22

bb20:		; preds = %bb18
	invoke void @__gnat_rcheck_12( i8* getelementptr ([14 x i8]* @.str1, i32 0, i32 0), i32 324 ) noreturn
			to label %invcont21 unwind label %lpad

invcont21:		; preds = %bb20
	unreachable

bb22:		; preds = %bb18
	ret i8 %iftmp.339.0

bb23:		; preds = %lpad
	call void @__gnat_begin_handler( i8* %eh_ptr ) nounwind
	%15 = load void ()** @system__soft_links__abort_undefer, align 4		; <void ()*> [#uses=1]
	invoke void %15( )
			to label %invcont24 unwind label %lpad33

invcont24:		; preds = %bb23
	invoke void @__gnat_raise_exception( %struct.system__standard_library__exception_data* bitcast (%struct.exception* @ada__io_exceptions__data_error to %struct.system__standard_library__exception_data*), i8* getelementptr ([47 x i8]* @.str2, i32 0, i32 0), %struct.string___XUB* @C.354.2200 ) noreturn
			to label %invcont27 unwind label %lpad33

invcont27:		; preds = %invcont24
	unreachable

lpad:		; preds = %bb20, %bb16, %bb8, %bb2, %entry
	%eh_ptr = call i8* @llvm.eh.exception( )		; <i8*> [#uses=4]
	%eh_select32 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), %struct.exception* @constraint_error, i32* @__gnat_all_others_value )		; <i32> [#uses=1]
	%eh_typeid = call i32 @llvm.eh.typeid.for.i32( i8* getelementptr (%struct.exception* @constraint_error, i32 0, i32 0) )		; <i32> [#uses=1]
	%16 = icmp eq i32 %eh_select32, %eh_typeid		; <i1> [#uses=1]
	br i1 %16, label %bb23, label %Unwind

lpad33:		; preds = %invcont24, %bb23
	%eh_ptr34 = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select36 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr34, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	call void @__gnat_end_handler( i8* %eh_ptr )
	br label %Unwind

Unwind:		; preds = %lpad, %lpad33
	%eh_exception.0 = phi i8* [ %eh_ptr, %lpad ], [ %eh_ptr34, %lpad33 ]		; <i8*> [#uses=1]
	%17 = call i32 (...)* @_Unwind_Resume( i8* %eh_exception.0 )		; <i32> [#uses=0]
	unreachable
}

define internal fastcc void @ce3806g__fxio__put.1149(%struct.ada__text_io__text_afcb* %file) {
entry:
	%A.301 = alloca %struct.string___XUB		; <%struct.string___XUB*> [#uses=3]
	%A.292 = alloca %struct.string___XUB		; <%struct.string___XUB*> [#uses=3]
	%0 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%1 = alloca [12 x i8]		; <[12 x i8]*> [#uses=1]
	%.sub = getelementptr [12 x i8]* %1, i32 0, i32 0		; <i8*> [#uses=2]
	%2 = getelementptr %struct.string___XUB* %A.292, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %2, align 8
	%3 = getelementptr %struct.string___XUB* %A.292, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 12, i32* %3, align 4
	%4 = invoke fastcc i32 @ce3806g__fxio__put__4.1215( i8* %.sub, %struct.string___XUB* %A.292, i8 signext -3 )
			to label %invcont unwind label %lpad		; <i32> [#uses=1]

invcont:		; preds = %entry
	%5 = getelementptr %struct.string___XUB* %A.301, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %5, align 8
	%6 = getelementptr %struct.string___XUB* %A.301, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %4, i32* %6, align 4
	invoke void @ada__text_io__generic_aux__put_item( %struct.ada__text_io__text_afcb* %file, i8* %.sub, %struct.string___XUB* %A.301 )
			to label %bb60 unwind label %lpad

bb60:		; preds = %invcont
	ret void

lpad:		; preds = %entry, %invcont
	%eh_ptr = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select62 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	call void @llvm.stackrestore( i8* %0 )
	%7 = call i32 (...)* @_Unwind_Resume( i8* %eh_ptr )		; <i32> [#uses=0]
	unreachable
}

define internal fastcc void @ce3806g__fxio__put__2.1155() {
entry:
	%A.266 = alloca %struct.string___XUB		; <%struct.string___XUB*> [#uses=3]
	%A.257 = alloca %struct.string___XUB		; <%struct.string___XUB*> [#uses=3]
	%0 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%1 = alloca [12 x i8]		; <[12 x i8]*> [#uses=1]
	%.sub = getelementptr [12 x i8]* %1, i32 0, i32 0		; <i8*> [#uses=2]
	%2 = getelementptr %struct.string___XUB* %A.257, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %2, align 8
	%3 = getelementptr %struct.string___XUB* %A.257, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 12, i32* %3, align 4
	%4 = invoke fastcc i32 @ce3806g__fxio__put__4.1215( i8* %.sub, %struct.string___XUB* %A.257, i8 signext -1 )
			to label %invcont unwind label %lpad		; <i32> [#uses=1]

invcont:		; preds = %entry
	%5 = getelementptr %struct.string___XUB* %A.266, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %5, align 8
	%6 = getelementptr %struct.string___XUB* %A.266, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %4, i32* %6, align 4
	%7 = load %struct.ada__text_io__text_afcb** @ada__text_io__current_out, align 4		; <%struct.ada__text_io__text_afcb*> [#uses=1]
	invoke void @ada__text_io__generic_aux__put_item( %struct.ada__text_io__text_afcb* %7, i8* %.sub, %struct.string___XUB* %A.266 )
			to label %bb60 unwind label %lpad

bb60:		; preds = %invcont
	ret void

lpad:		; preds = %entry, %invcont
	%eh_ptr = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select62 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	call void @llvm.stackrestore( i8* %0 )
	%8 = call i32 (...)* @_Unwind_Resume( i8* %eh_ptr )		; <i32> [#uses=0]
	unreachable
}

define internal fastcc i32 @ce3806g__fxio__put__4.1215(i8* %to.0, %struct.string___XUB* %to.1, i8 signext %item) {
entry:
        %P0 = load i32 * @__gnat_all_others_value, align 4  ; <i32*> [#uses=1]
        %P = alloca i32, i32 %P0	; <i32*> [#uses=1]
        call void @ext( i32* %P )
	%to_addr = alloca %struct.system__file_control_block__pstring		; <%struct.system__file_control_block__pstring*> [#uses=4]
	%FRAME.358 = alloca %struct.FRAME.ce3806g__fxio__put__4		; <%struct.FRAME.ce3806g__fxio__put__4*> [#uses=65]
	%0 = getelementptr %struct.system__file_control_block__pstring* %to_addr, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* %to.0, i8** %0, align 8
	%1 = getelementptr %struct.system__file_control_block__pstring* %to_addr, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	store %struct.string___XUB* %to.1, %struct.string___XUB** %1
	%2 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	store %struct.system__file_control_block__pstring* %to_addr, %struct.system__file_control_block__pstring** %2, align 4
	%3 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 3, i32* %3, align 8
	%4 = getelementptr %struct.system__file_control_block__pstring* %to_addr, i32 0, i32 1		; <%struct.string___XUB**> [#uses=1]
	%5 = load %struct.string___XUB** %4, align 4		; <%struct.string___XUB*> [#uses=1]
	%6 = getelementptr %struct.string___XUB* %5, i32 0, i32 0		; <i32*> [#uses=1]
	%7 = load i32* %6, align 4		; <i32> [#uses=1]
	%8 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 2		; <i32*> [#uses=1]
	store i32 %7, i32* %8, align 8
	%9 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 2		; <i32*> [#uses=1]
	%10 = load i32* %9, align 8		; <i32> [#uses=1]
	%11 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=1]
	store i32 %10, i32* %11, align 8
	%item.lobit = lshr i8 %item, 7		; <i8> [#uses=1]
	%12 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 6		; <i8*> [#uses=1]
	store i8 %item.lobit, i8* %12, align 8
	%13 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 2		; <i32*> [#uses=1]
	%14 = load i32* %13, align 8		; <i32> [#uses=1]
	%15 = add i32 %14, -1		; <i32> [#uses=1]
	%16 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %15, i32* %16, align 4
	%17 = sext i8 %item to i64		; <i64> [#uses=1]
	%18 = call i64 @system__exn_lli__exn_long_long_integer( i64 10, i32 1 ) readnone		; <i64> [#uses=1]
	%19 = sub i64 0, %18		; <i64> [#uses=1]
	%20 = call i64 @system__exn_lli__exn_long_long_integer( i64 10, i32 0 ) readnone		; <i64> [#uses=1]
	%21 = mul i64 %20, -2		; <i64> [#uses=1]
	call fastcc void @ce3806g__fxio__put__put_scaled__4.1346( %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i64 %17, i64 %19, i64 %21, i32 0, i32 -1 )
	%22 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%23 = load i32* %22, align 4		; <i32> [#uses=1]
	%24 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 2		; <i32*> [#uses=1]
	%25 = load i32* %24, align 8		; <i32> [#uses=1]
	%26 = icmp slt i32 %23, %25		; <i1> [#uses=1]
	br i1 %26, label %bb71, label %bb72

bb71:		; preds = %entry
	%27 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 0, i32* %27, align 4
	br label %bb72

bb72:		; preds = %entry, %bb102, %bb71
	%28 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=1]
	%29 = load i32* %28, align 4		; <i32> [#uses=1]
	%30 = icmp slt i32 %29, -1		; <i1> [#uses=1]
	%31 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%32 = load i32* %31, align 4		; <i32> [#uses=2]
	br i1 %30, label %bb103, label %bb74

bb74:		; preds = %bb72
	%33 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 2		; <i32*> [#uses=1]
	%34 = load i32* %33, align 8		; <i32> [#uses=1]
	%35 = add i32 %34, -1		; <i32> [#uses=1]
	%36 = icmp eq i32 %32, %35		; <i1> [#uses=1]
	%37 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=1]
	%38 = load i32* %37, align 4		; <i32> [#uses=2]
	br i1 %36, label %bb76, label %bb98

bb76:		; preds = %bb74
	%39 = icmp slt i32 %38, 1		; <i1> [#uses=1]
	br i1 %39, label %bb80, label %bb102

bb80:		; preds = %bb76
	%40 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=1]
	%41 = load i32* %40, align 4		; <i32> [#uses=2]
	%42 = icmp sgt i32 %41, -1		; <i1> [#uses=1]
	%.op = add i32 %41, 2		; <i32> [#uses=1]
	%43 = select i1 %42, i32 %.op, i32 2		; <i32> [#uses=1]
	%44 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 6		; <i8*> [#uses=1]
	%45 = load i8* %44, align 8		; <i8> [#uses=1]
	%46 = zext i8 %45 to i32		; <i32> [#uses=1]
	%47 = add i32 %43, %46		; <i32> [#uses=2]
	%48 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 0		; <i32*> [#uses=1]
	%49 = load i32* %48, align 8		; <i32> [#uses=1]
	%50 = icmp sgt i32 %47, %49		; <i1> [#uses=1]
	br i1 %50, label %bb88, label %bb85

bb85:		; preds = %bb80, %bb87
	%j.0 = phi i32 [ %68, %bb87 ], [ %47, %bb80 ]		; <i32> [#uses=2]
	%51 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%52 = load i32* %51, align 4		; <i32> [#uses=1]
	%53 = add i32 %52, 1		; <i32> [#uses=1]
	%54 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %53, i32* %54, align 4
	%55 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=1]
	%56 = load i32* %55, align 8		; <i32> [#uses=1]
	%57 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%58 = load %struct.system__file_control_block__pstring** %57, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%59 = getelementptr %struct.system__file_control_block__pstring* %58, i32 0, i32 0		; <i8**> [#uses=1]
	%60 = load i8** %59, align 4		; <i8*> [#uses=1]
	%61 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%62 = load i32* %61, align 4		; <i32> [#uses=1]
	%63 = sub i32 %62, %56		; <i32> [#uses=1]
	%64 = getelementptr i8* %60, i32 %63		; <i8*> [#uses=1]
	store i8 32, i8* %64, align 1
	%65 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 0		; <i32*> [#uses=1]
	%66 = load i32* %65, align 8		; <i32> [#uses=1]
	%67 = icmp eq i32 %66, %j.0		; <i1> [#uses=1]
	br i1 %67, label %bb88, label %bb87

bb87:		; preds = %bb85
	%68 = add i32 %j.0, 1		; <i32> [#uses=1]
	br label %bb85

bb88:		; preds = %bb80, %bb85
	%69 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 6		; <i8*> [#uses=1]
	%70 = load i8* %69, align 8		; <i8> [#uses=1]
	%toBool89 = icmp eq i8 %70, 0		; <i1> [#uses=1]
	br i1 %toBool89, label %bb91, label %bb90

bb90:		; preds = %bb88
	%71 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%72 = load i32* %71, align 4		; <i32> [#uses=1]
	%73 = add i32 %72, 1		; <i32> [#uses=1]
	%74 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %73, i32* %74, align 4
	%75 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=1]
	%76 = load i32* %75, align 8		; <i32> [#uses=1]
	%77 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%78 = load %struct.system__file_control_block__pstring** %77, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%79 = getelementptr %struct.system__file_control_block__pstring* %78, i32 0, i32 0		; <i8**> [#uses=1]
	%80 = load i8** %79, align 4		; <i8*> [#uses=1]
	%81 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%82 = load i32* %81, align 4		; <i32> [#uses=1]
	%83 = sub i32 %82, %76		; <i32> [#uses=1]
	%84 = getelementptr i8* %80, i32 %83		; <i8*> [#uses=1]
	store i8 45, i8* %84, align 1
	br label %bb91

bb91:		; preds = %bb88, %bb90
	%85 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=1]
	%86 = load i32* %85, align 4		; <i32> [#uses=1]
	%87 = icmp slt i32 %86, 0		; <i1> [#uses=1]
	br i1 %87, label %bb93, label %bb97

bb93:		; preds = %bb91
	%88 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%89 = load i32* %88, align 4		; <i32> [#uses=1]
	%90 = add i32 %89, 1		; <i32> [#uses=1]
	%91 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %90, i32* %91, align 4
	%92 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=1]
	%93 = load i32* %92, align 8		; <i32> [#uses=1]
	%94 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%95 = load %struct.system__file_control_block__pstring** %94, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%96 = getelementptr %struct.system__file_control_block__pstring* %95, i32 0, i32 0		; <i8**> [#uses=1]
	%97 = load i8** %96, align 4		; <i8*> [#uses=1]
	%98 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%99 = load i32* %98, align 4		; <i32> [#uses=1]
	%100 = sub i32 %99, %93		; <i32> [#uses=1]
	%101 = getelementptr i8* %97, i32 %100		; <i8*> [#uses=1]
	store i8 48, i8* %101, align 1
	%102 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%103 = load i32* %102, align 4		; <i32> [#uses=1]
	%104 = add i32 %103, 1		; <i32> [#uses=1]
	%105 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %104, i32* %105, align 4
	%106 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=1]
	%107 = load i32* %106, align 8		; <i32> [#uses=1]
	%108 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%109 = load %struct.system__file_control_block__pstring** %108, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%110 = getelementptr %struct.system__file_control_block__pstring* %109, i32 0, i32 0		; <i8**> [#uses=1]
	%111 = load i8** %110, align 4		; <i8*> [#uses=1]
	%112 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%113 = load i32* %112, align 4		; <i32> [#uses=1]
	%114 = sub i32 %113, %107		; <i32> [#uses=1]
	%115 = getelementptr i8* %111, i32 %114		; <i8*> [#uses=1]
	store i8 46, i8* %115, align 1
	%116 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=1]
	%117 = load i32* %116, align 4		; <i32> [#uses=1]
	br label %bb94

bb94:		; preds = %bb96, %bb93
	%j8.0 = phi i32 [ %117, %bb93 ], [ %133, %bb96 ]		; <i32> [#uses=2]
	%118 = icmp sgt i32 %j8.0, -2		; <i1> [#uses=1]
	br i1 %118, label %bb97, label %bb96

bb96:		; preds = %bb94
	%119 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%120 = load i32* %119, align 4		; <i32> [#uses=1]
	%121 = add i32 %120, 1		; <i32> [#uses=1]
	%122 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %121, i32* %122, align 4
	%123 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=1]
	%124 = load i32* %123, align 8		; <i32> [#uses=1]
	%125 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%126 = load %struct.system__file_control_block__pstring** %125, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%127 = getelementptr %struct.system__file_control_block__pstring* %126, i32 0, i32 0		; <i8**> [#uses=1]
	%128 = load i8** %127, align 4		; <i8*> [#uses=1]
	%129 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%130 = load i32* %129, align 4		; <i32> [#uses=1]
	%131 = sub i32 %130, %124		; <i32> [#uses=1]
	%132 = getelementptr i8* %128, i32 %131		; <i8*> [#uses=1]
	store i8 48, i8* %132, align 1
	%133 = add i32 %j8.0, 1		; <i32> [#uses=1]
	br label %bb94

bb97:		; preds = %bb91, %bb94
	%134 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%135 = load i32* %134, align 4		; <i32> [#uses=1]
	%136 = add i32 %135, 1		; <i32> [#uses=1]
	%137 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %136, i32* %137, align 4
	%138 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=1]
	%139 = load i32* %138, align 8		; <i32> [#uses=1]
	%140 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%141 = load %struct.system__file_control_block__pstring** %140, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%142 = getelementptr %struct.system__file_control_block__pstring* %141, i32 0, i32 0		; <i8**> [#uses=1]
	%143 = load i8** %142, align 4		; <i8*> [#uses=1]
	%144 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%145 = load i32* %144, align 4		; <i32> [#uses=1]
	%146 = sub i32 %145, %139		; <i32> [#uses=1]
	%147 = getelementptr i8* %143, i32 %146		; <i8*> [#uses=1]
	store i8 48, i8* %147, align 1
	br label %bb102

bb98:		; preds = %bb74
	%148 = icmp eq i32 %38, -1		; <i1> [#uses=1]
	br i1 %148, label %bb100, label %bb101

bb100:		; preds = %bb98
	%149 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%150 = load i32* %149, align 4		; <i32> [#uses=1]
	%151 = add i32 %150, 1		; <i32> [#uses=1]
	%152 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %151, i32* %152, align 4
	%153 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=1]
	%154 = load i32* %153, align 8		; <i32> [#uses=1]
	%155 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%156 = load %struct.system__file_control_block__pstring** %155, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%157 = getelementptr %struct.system__file_control_block__pstring* %156, i32 0, i32 0		; <i8**> [#uses=1]
	%158 = load i8** %157, align 4		; <i8*> [#uses=1]
	%159 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%160 = load i32* %159, align 4		; <i32> [#uses=1]
	%161 = sub i32 %160, %154		; <i32> [#uses=1]
	%162 = getelementptr i8* %158, i32 %161		; <i8*> [#uses=1]
	store i8 46, i8* %162, align 1
	br label %bb101

bb101:		; preds = %bb98, %bb100
	%163 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%164 = load i32* %163, align 4		; <i32> [#uses=1]
	%165 = add i32 %164, 1		; <i32> [#uses=1]
	%166 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %165, i32* %166, align 4
	%167 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 4		; <i32*> [#uses=1]
	%168 = load i32* %167, align 8		; <i32> [#uses=1]
	%169 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%170 = load %struct.system__file_control_block__pstring** %169, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%171 = getelementptr %struct.system__file_control_block__pstring* %170, i32 0, i32 0		; <i8**> [#uses=1]
	%172 = load i8** %171, align 4		; <i8*> [#uses=1]
	%173 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 5		; <i32*> [#uses=1]
	%174 = load i32* %173, align 4		; <i32> [#uses=1]
	%175 = sub i32 %174, %168		; <i32> [#uses=1]
	%176 = getelementptr i8* %172, i32 %175		; <i8*> [#uses=1]
	store i8 48, i8* %176, align 1
	br label %bb102

bb102:		; preds = %bb76, %bb101, %bb97
	%177 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=1]
	%178 = load i32* %177, align 4		; <i32> [#uses=1]
	%179 = add i32 %178, -1		; <i32> [#uses=1]
	%180 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %FRAME.358, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %179, i32* %180, align 4
	br label %bb72

bb103:		; preds = %bb72
	ret i32 %32
}

declare x86_fp80 @ada__text_io__float_aux__get(%struct.ada__text_io__text_afcb*, i32)

declare void @__gnat_rcheck_12(i8*, i32) noreturn

declare void @__gnat_rcheck_10(i8*, i32) noreturn

declare i8* @llvm.eh.exception() nounwind

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...) nounwind

declare i32 @llvm.eh.typeid.for.i32(i8*) nounwind

declare void @__gnat_begin_handler(i8*) nounwind

declare void @__gnat_raise_exception(%struct.system__standard_library__exception_data*, i8*, %struct.string___XUB*) noreturn

declare void @__gnat_end_handler(i8*)

declare i32 @__gnat_eh_personality(...)

declare i32 @_Unwind_Resume(...)

define internal fastcc void @ce3806g__fxio__put__put_int64__4.1339(%struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i64 %x, i32 %scale) {
entry:
	%0 = icmp eq i64 %x, 0		; <i1> [#uses=1]
	br i1 %0, label %return, label %bb

bb:		; preds = %entry
	%1 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %scale, i32* %1, align 4
	%2 = add i64 %x, 9		; <i64> [#uses=1]
	%3 = icmp ugt i64 %2, 18		; <i1> [#uses=1]
	br i1 %3, label %bb18, label %bb19

bb18:		; preds = %bb
	%4 = add i32 %scale, 1		; <i32> [#uses=1]
	%5 = sdiv i64 %x, 10		; <i64> [#uses=1]
	call fastcc void @ce3806g__fxio__put__put_int64__4.1339( %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i64 %5, i32 %4 )
	br label %bb19

bb19:		; preds = %bb, %bb18
	%6 = srem i64 %x, 10		; <i64> [#uses=3]
	%neg = sub i64 0, %6		; <i64> [#uses=1]
	%abscond = icmp sgt i64 %6, -1		; <i1> [#uses=1]
	%abs = select i1 %abscond, i64 %6, i64 %neg		; <i64> [#uses=3]
	%7 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%8 = load i32* %7, align 4		; <i32> [#uses=1]
	%9 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 2		; <i32*> [#uses=1]
	%10 = load i32* %9, align 4		; <i32> [#uses=1]
	%11 = add i32 %10, -1		; <i32> [#uses=1]
	%12 = icmp eq i32 %8, %11		; <i1> [#uses=1]
	br i1 %12, label %bb23, label %bb44

bb23:		; preds = %bb19
	%13 = icmp ne i64 %abs, 0		; <i1> [#uses=1]
	%14 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 1		; <i32*> [#uses=1]
	%15 = load i32* %14, align 4		; <i32> [#uses=1]
	%16 = icmp slt i32 %15, 1		; <i1> [#uses=1]
	%17 = or i1 %13, %16		; <i1> [#uses=1]
	br i1 %17, label %bb27, label %bb48

bb27:		; preds = %bb23
	%18 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 1		; <i32*> [#uses=1]
	%19 = load i32* %18, align 4		; <i32> [#uses=2]
	%20 = icmp sgt i32 %19, -1		; <i1> [#uses=1]
	%.op = add i32 %19, 2		; <i32> [#uses=1]
	%21 = select i1 %20, i32 %.op, i32 2		; <i32> [#uses=1]
	%22 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 6		; <i8*> [#uses=1]
	%23 = load i8* %22, align 1		; <i8> [#uses=1]
	%24 = zext i8 %23 to i32		; <i32> [#uses=1]
	%25 = add i32 %21, %24		; <i32> [#uses=2]
	%26 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 0		; <i32*> [#uses=1]
	%27 = load i32* %26, align 4		; <i32> [#uses=1]
	%28 = icmp sgt i32 %25, %27		; <i1> [#uses=1]
	br i1 %28, label %bb34, label %bb31

bb31:		; preds = %bb27, %bb33
	%j.0 = phi i32 [ %46, %bb33 ], [ %25, %bb27 ]		; <i32> [#uses=2]
	%29 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%30 = load i32* %29, align 4		; <i32> [#uses=1]
	%31 = add i32 %30, 1		; <i32> [#uses=1]
	%32 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %31, i32* %32, align 4
	%33 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%34 = load i32* %33, align 4		; <i32> [#uses=1]
	%35 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%36 = load %struct.system__file_control_block__pstring** %35, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%37 = getelementptr %struct.system__file_control_block__pstring* %36, i32 0, i32 0		; <i8**> [#uses=1]
	%38 = load i8** %37, align 4		; <i8*> [#uses=1]
	%39 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%40 = load i32* %39, align 4		; <i32> [#uses=1]
	%41 = sub i32 %40, %34		; <i32> [#uses=1]
	%42 = getelementptr i8* %38, i32 %41		; <i8*> [#uses=1]
	store i8 32, i8* %42, align 1
	%43 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 0		; <i32*> [#uses=1]
	%44 = load i32* %43, align 4		; <i32> [#uses=1]
	%45 = icmp eq i32 %44, %j.0		; <i1> [#uses=1]
	br i1 %45, label %bb34, label %bb33

bb33:		; preds = %bb31
	%46 = add i32 %j.0, 1		; <i32> [#uses=1]
	br label %bb31

bb34:		; preds = %bb27, %bb31
	%47 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 6		; <i8*> [#uses=1]
	%48 = load i8* %47, align 1		; <i8> [#uses=1]
	%toBool35 = icmp eq i8 %48, 0		; <i1> [#uses=1]
	br i1 %toBool35, label %bb37, label %bb36

bb36:		; preds = %bb34
	%49 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%50 = load i32* %49, align 4		; <i32> [#uses=1]
	%51 = add i32 %50, 1		; <i32> [#uses=1]
	%52 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %51, i32* %52, align 4
	%53 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%54 = load i32* %53, align 4		; <i32> [#uses=1]
	%55 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%56 = load %struct.system__file_control_block__pstring** %55, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%57 = getelementptr %struct.system__file_control_block__pstring* %56, i32 0, i32 0		; <i8**> [#uses=1]
	%58 = load i8** %57, align 4		; <i8*> [#uses=1]
	%59 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%60 = load i32* %59, align 4		; <i32> [#uses=1]
	%61 = sub i32 %60, %54		; <i32> [#uses=1]
	%62 = getelementptr i8* %58, i32 %61		; <i8*> [#uses=1]
	store i8 45, i8* %62, align 1
	br label %bb37

bb37:		; preds = %bb34, %bb36
	%63 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 1		; <i32*> [#uses=1]
	%64 = load i32* %63, align 4		; <i32> [#uses=1]
	%65 = icmp slt i32 %64, 0		; <i1> [#uses=1]
	br i1 %65, label %bb39, label %bb43

bb39:		; preds = %bb37
	%66 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%67 = load i32* %66, align 4		; <i32> [#uses=1]
	%68 = add i32 %67, 1		; <i32> [#uses=1]
	%69 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %68, i32* %69, align 4
	%70 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%71 = load i32* %70, align 4		; <i32> [#uses=1]
	%72 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%73 = load %struct.system__file_control_block__pstring** %72, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%74 = getelementptr %struct.system__file_control_block__pstring* %73, i32 0, i32 0		; <i8**> [#uses=1]
	%75 = load i8** %74, align 4		; <i8*> [#uses=1]
	%76 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%77 = load i32* %76, align 4		; <i32> [#uses=1]
	%78 = sub i32 %77, %71		; <i32> [#uses=1]
	%79 = getelementptr i8* %75, i32 %78		; <i8*> [#uses=1]
	store i8 48, i8* %79, align 1
	%80 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%81 = load i32* %80, align 4		; <i32> [#uses=1]
	%82 = add i32 %81, 1		; <i32> [#uses=1]
	%83 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %82, i32* %83, align 4
	%84 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%85 = load i32* %84, align 4		; <i32> [#uses=1]
	%86 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%87 = load %struct.system__file_control_block__pstring** %86, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%88 = getelementptr %struct.system__file_control_block__pstring* %87, i32 0, i32 0		; <i8**> [#uses=1]
	%89 = load i8** %88, align 4		; <i8*> [#uses=1]
	%90 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%91 = load i32* %90, align 4		; <i32> [#uses=1]
	%92 = sub i32 %91, %85		; <i32> [#uses=1]
	%93 = getelementptr i8* %89, i32 %92		; <i8*> [#uses=1]
	store i8 46, i8* %93, align 1
	%94 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 1		; <i32*> [#uses=1]
	%95 = load i32* %94, align 4		; <i32> [#uses=1]
	br label %bb40

bb40:		; preds = %bb42, %bb39
	%j15.0 = phi i32 [ %95, %bb39 ], [ %111, %bb42 ]		; <i32> [#uses=2]
	%96 = icmp sgt i32 %j15.0, -2		; <i1> [#uses=1]
	br i1 %96, label %bb43, label %bb42

bb42:		; preds = %bb40
	%97 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%98 = load i32* %97, align 4		; <i32> [#uses=1]
	%99 = add i32 %98, 1		; <i32> [#uses=1]
	%100 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %99, i32* %100, align 4
	%101 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%102 = load i32* %101, align 4		; <i32> [#uses=1]
	%103 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%104 = load %struct.system__file_control_block__pstring** %103, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%105 = getelementptr %struct.system__file_control_block__pstring* %104, i32 0, i32 0		; <i8**> [#uses=1]
	%106 = load i8** %105, align 4		; <i8*> [#uses=1]
	%107 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%108 = load i32* %107, align 4		; <i32> [#uses=1]
	%109 = sub i32 %108, %102		; <i32> [#uses=1]
	%110 = getelementptr i8* %106, i32 %109		; <i8*> [#uses=1]
	store i8 48, i8* %110, align 1
	%111 = add i32 %j15.0, 1		; <i32> [#uses=1]
	br label %bb40

bb43:		; preds = %bb37, %bb40
	%112 = trunc i64 %abs to i32		; <i32> [#uses=1]
	%113 = getelementptr [10 x i8]* @.str3, i32 0, i32 %112		; <i8*> [#uses=1]
	%114 = load i8* %113, align 1		; <i8> [#uses=1]
	%115 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%116 = load i32* %115, align 4		; <i32> [#uses=1]
	%117 = add i32 %116, 1		; <i32> [#uses=1]
	%118 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %117, i32* %118, align 4
	%119 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%120 = load i32* %119, align 4		; <i32> [#uses=1]
	%121 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%122 = load %struct.system__file_control_block__pstring** %121, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%123 = getelementptr %struct.system__file_control_block__pstring* %122, i32 0, i32 0		; <i8**> [#uses=1]
	%124 = load i8** %123, align 4		; <i8*> [#uses=1]
	%125 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%126 = load i32* %125, align 4		; <i32> [#uses=1]
	%127 = sub i32 %126, %120		; <i32> [#uses=1]
	%128 = getelementptr i8* %124, i32 %127		; <i8*> [#uses=1]
	store i8 %114, i8* %128, align 1
	br label %bb48

bb44:		; preds = %bb19
	%129 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 1		; <i32*> [#uses=1]
	%130 = load i32* %129, align 4		; <i32> [#uses=1]
	%131 = icmp eq i32 %130, -1		; <i1> [#uses=1]
	br i1 %131, label %bb46, label %bb47

bb46:		; preds = %bb44
	%132 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%133 = load i32* %132, align 4		; <i32> [#uses=1]
	%134 = add i32 %133, 1		; <i32> [#uses=1]
	%135 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %134, i32* %135, align 4
	%136 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%137 = load i32* %136, align 4		; <i32> [#uses=1]
	%138 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%139 = load %struct.system__file_control_block__pstring** %138, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%140 = getelementptr %struct.system__file_control_block__pstring* %139, i32 0, i32 0		; <i8**> [#uses=1]
	%141 = load i8** %140, align 4		; <i8*> [#uses=1]
	%142 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%143 = load i32* %142, align 4		; <i32> [#uses=1]
	%144 = sub i32 %143, %137		; <i32> [#uses=1]
	%145 = getelementptr i8* %141, i32 %144		; <i8*> [#uses=1]
	store i8 46, i8* %145, align 1
	br label %bb47

bb47:		; preds = %bb44, %bb46
	%146 = trunc i64 %abs to i32		; <i32> [#uses=1]
	%147 = getelementptr [10 x i8]* @.str3, i32 0, i32 %146		; <i8*> [#uses=1]
	%148 = load i8* %147, align 1		; <i8> [#uses=1]
	%149 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%150 = load i32* %149, align 4		; <i32> [#uses=1]
	%151 = add i32 %150, 1		; <i32> [#uses=1]
	%152 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	store i32 %151, i32* %152, align 4
	%153 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 4		; <i32*> [#uses=1]
	%154 = load i32* %153, align 4		; <i32> [#uses=1]
	%155 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 3		; <%struct.system__file_control_block__pstring**> [#uses=1]
	%156 = load %struct.system__file_control_block__pstring** %155, align 4		; <%struct.system__file_control_block__pstring*> [#uses=1]
	%157 = getelementptr %struct.system__file_control_block__pstring* %156, i32 0, i32 0		; <i8**> [#uses=1]
	%158 = load i8** %157, align 4		; <i8*> [#uses=1]
	%159 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 5		; <i32*> [#uses=1]
	%160 = load i32* %159, align 4		; <i32> [#uses=1]
	%161 = sub i32 %160, %154		; <i32> [#uses=1]
	%162 = getelementptr i8* %158, i32 %161		; <i8*> [#uses=1]
	store i8 %148, i8* %162, align 1
	br label %bb48

bb48:		; preds = %bb23, %bb47, %bb43
	%163 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 1		; <i32*> [#uses=1]
	%164 = load i32* %163, align 4		; <i32> [#uses=1]
	%165 = add i32 %164, -1		; <i32> [#uses=1]
	%166 = getelementptr %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.361, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 %165, i32* %166, align 4
	ret void

return:		; preds = %entry
	ret void
}

define internal fastcc void @ce3806g__fxio__put__put_scaled__4.1346(%struct.FRAME.ce3806g__fxio__put__4* %CHAIN.365, i64 %x, i64 %y, i64 %z, i32 %a, i32 %e) {
entry:
	%0 = alloca { i64, i64 }		; <{ i64, i64 }*> [#uses=3]
	%1 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%2 = add i32 %a, 17		; <i32> [#uses=2]
	%3 = sdiv i32 %2, 18		; <i32> [#uses=3]
	%4 = add i32 %3, 1		; <i32> [#uses=7]
	%5 = icmp sgt i32 %4, -1		; <i1> [#uses=1]
	%max53 = select i1 %5, i32 %4, i32 0		; <i32> [#uses=1]
	%6 = alloca i64, i32 %max53		; <i64*> [#uses=21]
	%7 = icmp sgt i32 %4, 0		; <i1> [#uses=1]
	br i1 %7, label %bb55, label %bb58

bb55:		; preds = %entry, %bb57
	%J60b.0 = phi i32 [ %11, %bb57 ], [ 1, %entry ]		; <i32> [#uses=3]
	%8 = add i32 %J60b.0, -1		; <i32> [#uses=1]
	%9 = getelementptr i64* %6, i32 %8		; <i64*> [#uses=1]
	store i64 0, i64* %9, align 8
	%10 = icmp eq i32 %4, %J60b.0		; <i1> [#uses=1]
	br i1 %10, label %bb58, label %bb57

bb57:		; preds = %bb55
	%11 = add i32 %J60b.0, 1		; <i32> [#uses=1]
	br label %bb55

bb58:		; preds = %entry, %bb55
	%12 = icmp sgt i32 %4, 0		; <i1> [#uses=1]
	br i1 %12, label %bb61, label %bb91

bb61:		; preds = %bb58, %bb90
	%j2.0 = phi i32 [ %88, %bb90 ], [ 1, %bb58 ]		; <i32> [#uses=11]
	%aa.0 = phi i32 [ %86, %bb90 ], [ %a, %bb58 ]		; <i32> [#uses=6]
	%yy.0 = phi i64 [ %84, %bb90 ], [ %y, %bb58 ]		; <i64> [#uses=3]
	%xx.0 = phi i64 [ %21, %bb90 ], [ %x, %bb58 ]		; <i64> [#uses=2]
	%13 = icmp eq i64 %xx.0, 0		; <i1> [#uses=1]
	br i1 %13, label %bb91, label %bb63

bb63:		; preds = %bb61
	%14 = icmp eq i32 %aa.0, 0		; <i1> [#uses=1]
	%15 = zext i1 %14 to i8		; <i8> [#uses=1]
	invoke void @system__arith_64__scaled_divide( { i64, i64 }* noalias sret %0, i64 %xx.0, i64 %yy.0, i64 %z, i8 %15 )
			to label %invcont unwind label %lpad

invcont:		; preds = %bb63
	%16 = getelementptr { i64, i64 }* %0, i32 0, i32 0		; <i64*> [#uses=1]
	%17 = load i64* %16, align 8		; <i64> [#uses=1]
	%18 = add i32 %j2.0, -1		; <i32> [#uses=1]
	%19 = getelementptr i64* %6, i32 %18		; <i64*> [#uses=1]
	store i64 %17, i64* %19, align 8
	%20 = getelementptr { i64, i64 }* %0, i32 0, i32 1		; <i64*> [#uses=1]
	%21 = load i64* %20, align 8		; <i64> [#uses=1]
	%22 = add i32 %j2.0, -1		; <i32> [#uses=1]
	%23 = getelementptr i64* %6, i32 %22		; <i64*> [#uses=1]
	%24 = load i64* %23, align 8		; <i64> [#uses=1]
	%25 = icmp eq i64 %24, %yy.0		; <i1> [#uses=1]
	%26 = add i32 %j2.0, -1		; <i32> [#uses=1]
	%27 = getelementptr i64* %6, i32 %26		; <i64*> [#uses=1]
	%28 = load i64* %27, align 8		; <i64> [#uses=1]
	%29 = sub i64 0, %28		; <i64> [#uses=1]
	%30 = icmp eq i64 %yy.0, %29		; <i1> [#uses=1]
	%31 = or i1 %25, %30		; <i1> [#uses=1]
	%32 = icmp sgt i32 %j2.0, 1		; <i1> [#uses=1]
	%or.cond = and i1 %31, %32		; <i1> [#uses=1]
	br i1 %or.cond, label %bb69, label %bb83

bb69:		; preds = %invcont
	%33 = add i32 %j2.0, -1		; <i32> [#uses=1]
	%34 = getelementptr i64* %6, i32 %33		; <i64*> [#uses=1]
	%35 = load i64* %34, align 8		; <i64> [#uses=1]
	%36 = icmp slt i64 %35, 0		; <i1> [#uses=1]
	%37 = add i32 %j2.0, -2		; <i32> [#uses=1]
	%38 = getelementptr i64* %6, i32 %37		; <i64*> [#uses=1]
	%39 = load i64* %38, align 8		; <i64> [#uses=2]
	br i1 %36, label %bb71, label %bb72

bb71:		; preds = %bb69
	%40 = add i64 %39, 1		; <i64> [#uses=1]
	%41 = add i32 %j2.0, -2		; <i32> [#uses=1]
	%42 = getelementptr i64* %6, i32 %41		; <i64*> [#uses=1]
	store i64 %40, i64* %42, align 8
	br label %bb73

bb72:		; preds = %bb69
	%43 = add i64 %39, -1		; <i64> [#uses=1]
	%44 = add i32 %j2.0, -2		; <i32> [#uses=1]
	%45 = getelementptr i64* %6, i32 %44		; <i64*> [#uses=1]
	store i64 %43, i64* %45, align 8
	br label %bb73

bb73:		; preds = %bb72, %bb71
	%46 = add i32 %j2.0, -1		; <i32> [#uses=1]
	%47 = getelementptr i64* %6, i32 %46		; <i64*> [#uses=1]
	store i64 0, i64* %47, align 8
	br label %bb74

bb74:		; preds = %bb82, %bb73
	%j1.0 = phi i32 [ %4, %bb73 ], [ %81, %bb82 ]		; <i32> [#uses=12]
	%48 = icmp slt i32 %j1.0, 2		; <i1> [#uses=1]
	br i1 %48, label %bb83, label %bb76

bb76:		; preds = %bb74
	%49 = add i32 %j1.0, -1		; <i32> [#uses=1]
	%50 = getelementptr i64* %6, i32 %49		; <i64*> [#uses=1]
	%51 = load i64* %50, align 8		; <i64> [#uses=1]
	%52 = icmp sgt i64 %51, 999999999999999999		; <i1> [#uses=1]
	br i1 %52, label %bb78, label %bb79

bb78:		; preds = %bb76
	%53 = add i32 %j1.0, -2		; <i32> [#uses=1]
	%54 = getelementptr i64* %6, i32 %53		; <i64*> [#uses=1]
	%55 = load i64* %54, align 8		; <i64> [#uses=1]
	%56 = add i64 %55, 1		; <i64> [#uses=1]
	%57 = add i32 %j1.0, -2		; <i32> [#uses=1]
	%58 = getelementptr i64* %6, i32 %57		; <i64*> [#uses=1]
	store i64 %56, i64* %58, align 8
	%59 = add i32 %j1.0, -1		; <i32> [#uses=1]
	%60 = getelementptr i64* %6, i32 %59		; <i64*> [#uses=1]
	%61 = load i64* %60, align 8		; <i64> [#uses=1]
	%62 = add i64 %61, -1000000000000000000		; <i64> [#uses=1]
	%63 = add i32 %j1.0, -1		; <i32> [#uses=1]
	%64 = getelementptr i64* %6, i32 %63		; <i64*> [#uses=1]
	store i64 %62, i64* %64, align 8
	br label %bb82

bb79:		; preds = %bb76
	%65 = add i32 %j1.0, -1		; <i32> [#uses=1]
	%66 = getelementptr i64* %6, i32 %65		; <i64*> [#uses=1]
	%67 = load i64* %66, align 8		; <i64> [#uses=1]
	%68 = icmp slt i64 %67, -999999999999999999		; <i1> [#uses=1]
	br i1 %68, label %bb81, label %bb82

bb81:		; preds = %bb79
	%69 = add i32 %j1.0, -2		; <i32> [#uses=1]
	%70 = getelementptr i64* %6, i32 %69		; <i64*> [#uses=1]
	%71 = load i64* %70, align 8		; <i64> [#uses=1]
	%72 = add i64 %71, -1		; <i64> [#uses=1]
	%73 = add i32 %j1.0, -2		; <i32> [#uses=1]
	%74 = getelementptr i64* %6, i32 %73		; <i64*> [#uses=1]
	store i64 %72, i64* %74, align 8
	%75 = add i32 %j1.0, -1		; <i32> [#uses=1]
	%76 = getelementptr i64* %6, i32 %75		; <i64*> [#uses=1]
	%77 = load i64* %76, align 8		; <i64> [#uses=1]
	%78 = add i64 %77, 1000000000000000000		; <i64> [#uses=1]
	%79 = add i32 %j1.0, -1		; <i32> [#uses=1]
	%80 = getelementptr i64* %6, i32 %79		; <i64*> [#uses=1]
	store i64 %78, i64* %80, align 8
	br label %bb82

bb82:		; preds = %bb79, %bb81, %bb78
	%81 = add i32 %j1.0, -1		; <i32> [#uses=1]
	br label %bb74

bb83:		; preds = %invcont, %bb74
	%82 = icmp slt i32 %aa.0, 19		; <i1> [#uses=1]
	%min = select i1 %82, i32 %aa.0, i32 18		; <i32> [#uses=1]
	%83 = invoke i64 @system__exn_lli__exn_long_long_integer( i64 10, i32 %min ) readnone
			to label %invcont86 unwind label %lpad		; <i64> [#uses=1]

invcont86:		; preds = %bb83
	%84 = sub i64 0, %83		; <i64> [#uses=1]
	%85 = icmp slt i32 %aa.0, 19		; <i1> [#uses=1]
	%min87 = select i1 %85, i32 %aa.0, i32 18		; <i32> [#uses=1]
	%86 = sub i32 %aa.0, %min87		; <i32> [#uses=1]
	%87 = icmp eq i32 %4, %j2.0		; <i1> [#uses=1]
	br i1 %87, label %bb91, label %bb90

bb90:		; preds = %invcont86
	%88 = add i32 %j2.0, 1		; <i32> [#uses=1]
	br label %bb61

bb91:		; preds = %bb58, %bb61, %invcont86
	%89 = icmp slt i32 %2, 18		; <i1> [#uses=1]
	br i1 %89, label %bb98, label %bb94

bb94:		; preds = %bb91, %bb97
	%j.0 = phi i32 [ %97, %bb97 ], [ 1, %bb91 ]		; <i32> [#uses=4]
	%90 = mul i32 %j.0, 18		; <i32> [#uses=1]
	%91 = add i32 %90, -18		; <i32> [#uses=1]
	%92 = sub i32 %e, %91		; <i32> [#uses=1]
	%93 = add i32 %j.0, -1		; <i32> [#uses=1]
	%94 = getelementptr i64* %6, i32 %93		; <i64*> [#uses=1]
	%95 = load i64* %94, align 8		; <i64> [#uses=1]
	invoke fastcc void @ce3806g__fxio__put__put_int64__4.1339( %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.365, i64 %95, i32 %92 )
			to label %invcont95 unwind label %lpad

invcont95:		; preds = %bb94
	%96 = icmp eq i32 %3, %j.0		; <i1> [#uses=1]
	br i1 %96, label %bb98, label %bb97

bb97:		; preds = %invcont95
	%97 = add i32 %j.0, 1		; <i32> [#uses=1]
	br label %bb94

bb98:		; preds = %bb91, %invcont95
	%98 = sub i32 %e, %a		; <i32> [#uses=1]
	%99 = getelementptr i64* %6, i32 %3		; <i64*> [#uses=1]
	%100 = load i64* %99, align 8		; <i64> [#uses=1]
	invoke fastcc void @ce3806g__fxio__put__put_int64__4.1339( %struct.FRAME.ce3806g__fxio__put__4* %CHAIN.365, i64 %100, i32 %98 )
			to label %bb101 unwind label %lpad

bb101:		; preds = %bb98
	ret void

lpad:		; preds = %bb98, %bb94, %bb83, %bb63
	%eh_ptr = call i8* @llvm.eh.exception( )		; <i8*> [#uses=2]
	%eh_select103 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i32* @__gnat_all_others_value )		; <i32> [#uses=0]
	call void @llvm.stackrestore( i8* %1 )
	%101 = call i32 (...)* @_Unwind_Resume( i8* %eh_ptr )		; <i32> [#uses=0]
	unreachable
}

declare i8* @llvm.stacksave() nounwind

declare void @system__arith_64__scaled_divide({ i64, i64 }* noalias sret, i64, i64, i64, i8)

declare i64 @system__exn_lli__exn_long_long_integer(i64, i32) readnone

declare void @llvm.stackrestore(i8*) nounwind

declare i32 @system__img_real__set_image_real(x86_fp80, i8*, %struct.string___XUB*, i32, i32, i32, i32)

declare void @ada__text_io__generic_aux__put_item(%struct.ada__text_io__text_afcb*, i8*, %struct.string___XUB*)

declare void @report__test(i8*, %struct.string___XUB*, i8*, %struct.string___XUB*)

declare void @system__secondary_stack__ss_mark(%struct.string___XUB* noalias sret)

declare void @system__exception_table__register_exception(%struct.system__standard_library__exception_data*)

declare void @report__legal_file_name(%struct.system__file_control_block__pstring* noalias sret, i32, i8*, %struct.string___XUB*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__create(%struct.ada__text_io__text_afcb*, i8, i8*, %struct.string___XUB*, i8*, %struct.string___XUB*)

declare void @system__secondary_stack__ss_release(i32, i32)

declare void @report__not_applicable(i8*, %struct.string___XUB*)

declare void @ada__text_io__set_output(%struct.ada__text_io__text_afcb*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__close(%struct.ada__text_io__text_afcb*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__open(%struct.ada__text_io__text_afcb*, i8, i8*, %struct.string___XUB*, i8*, %struct.string___XUB*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__standard_output()

declare void @report__failed(i8*, %struct.string___XUB*)

declare void @ext(i32*)

declare %struct.ada__text_io__text_afcb* @ada__text_io__delete(%struct.ada__text_io__text_afcb*)

declare void @report__result()
