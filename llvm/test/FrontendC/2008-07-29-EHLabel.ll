; RUN: llvm-as < %s | llc -o - | %llvmgcc -xassembler -c -o /dev/null -
; PR2609
	%struct..0._11 = type { i32 }
	%struct..1__pthread_mutex_s = type { i32, i32, i32, i32, i32, %struct..0._11 }
	%struct.pthread_attr_t = type { i32, [32 x i8] }
	%struct.pthread_mutex_t = type { %struct..1__pthread_mutex_s }
	%"struct.std::__ctype_abstract_base<wchar_t>" = type { %"struct.std::locale::facet" }
	%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8, i8, %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, %"struct.std::ctype<char>"*, %"struct.std::__ctype_abstract_base<wchar_t>"*, %"struct.std::__ctype_abstract_base<wchar_t>"* }
	%"struct.std::basic_istream<char,std::char_traits<char> >" = type { i32 (...)**, i32, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_istream<char,std::char_traits<char> >::sentry" = type { i8 }
	%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { i32 (...)**, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"struct.std::locale" }
	%"struct.std::ctype<char>" = type { %"struct.std::locale::facet", i32*, i8, i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8 }
	%"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"struct.std::locale" }
	%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
	%"struct.std::ios_base::_Words" = type { i8*, i32 }
	%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
	%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
	%"struct.std::locale::facet" = type { i32 (...)**, i32 }

@_ZL20__gthrw_pthread_oncePiPFvvE = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]
@_ZL27__gthrw_pthread_getspecificj = alias weak i8* (i32)* @pthread_getspecific		; <i8* (i32)*> [#uses=0]
@_ZL27__gthrw_pthread_setspecificjPKv = alias weak i32 (i32, i8*)* @pthread_setspecific		; <i32 (i32, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = alias weak i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create		; <i32 (i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)*> [#uses=0]
@_ZL22__gthrw_pthread_cancelm = alias weak i32 (i32)* @pthread_cancel		; <i32 (i32)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_lock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_trylock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = alias weak i32 (%struct.pthread_mutex_t*)* @pthread_mutex_unlock		; <i32 (%struct.pthread_mutex_t*)*> [#uses=0]
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = alias weak i32 (%struct.pthread_mutex_t*, %struct..0._11*)* @pthread_mutex_init		; <i32 (%struct.pthread_mutex_t*, %struct..0._11*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_createPjPFvPvE = alias weak i32 (i32*, void (i8*)*)* @pthread_key_create		; <i32 (i32*, void (i8*)*)*> [#uses=0]
@_ZL26__gthrw_pthread_key_deletej = alias weak i32 (i32)* @pthread_key_delete		; <i32 (i32)*> [#uses=0]
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = alias weak i32 (%struct..0._11*)* @pthread_mutexattr_init		; <i32 (%struct..0._11*)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = alias weak i32 (%struct..0._11*, i32)* @pthread_mutexattr_settype		; <i32 (%struct..0._11*, i32)*> [#uses=0]
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = alias weak i32 (%struct..0._11*)* @pthread_mutexattr_destroy		; <i32 (%struct..0._11*)*> [#uses=0]

define %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSi7getlineEPcic(%"struct.std::basic_istream<char,std::char_traits<char> >"* %this, i8* %__s, i32 %__n, i8 signext  %__delim) {
entry:
	%__cerb = alloca %"struct.std::basic_istream<char,std::char_traits<char> >::sentry"		; <%"struct.std::basic_istream<char,std::char_traits<char> >::sentry"*> [#uses=2]
	getelementptr %"struct.std::basic_istream<char,std::char_traits<char> >"* %this, i32 0, i32 1		; <i32*>:0 [#uses=7]
	store i32 0, i32* %0, align 4
	call void @_ZNSi6sentryC1ERSib( %"struct.std::basic_istream<char,std::char_traits<char> >::sentry"* %__cerb, %"struct.std::basic_istream<char,std::char_traits<char> >"* %this, i8 zeroext  1 )
	getelementptr %"struct.std::basic_istream<char,std::char_traits<char> >::sentry"* %__cerb, i32 0, i32 0		; <i8*>:1 [#uses=1]
	load i8* %1, align 8		; <i8>:2 [#uses=1]
	%toBool = icmp eq i8 %2, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb162, label %bb

bb:		; preds = %entry
	zext i8 %__delim to i32		; <i32>:3 [#uses=1]
	getelementptr %"struct.std::basic_istream<char,std::char_traits<char> >"* %this, i32 0, i32 0		; <i32 (...)***>:4 [#uses=1]
	load i32 (...)*** %4, align 4		; <i32 (...)**>:5 [#uses=1]
	getelementptr i32 (...)** %5, i32 -3		; <i32 (...)**>:6 [#uses=1]
	bitcast i32 (...)** %6 to i32*		; <i32*>:7 [#uses=1]
	load i32* %7, align 4		; <i32>:8 [#uses=1]
	bitcast %"struct.std::basic_istream<char,std::char_traits<char> >"* %this to i8*		; <i8*>:9 [#uses=1]
	%ctg2186 = getelementptr i8* %9, i32 %8		; <i8*> [#uses=1]
	bitcast i8* %ctg2186 to %"struct.std::basic_ios<char,std::char_traits<char> >"*		; <%"struct.std::basic_ios<char,std::char_traits<char> >"*>:10 [#uses=1]
	getelementptr %"struct.std::basic_ios<char,std::char_traits<char> >"* %10, i32 0, i32 4		; <%"struct.std::basic_streambuf<char,std::char_traits<char> >"**>:11 [#uses=1]
	load %"struct.std::basic_streambuf<char,std::char_traits<char> >"** %11, align 4		; <%"struct.std::basic_streambuf<char,std::char_traits<char> >"*>:12 [#uses=9]
	getelementptr %"struct.std::basic_streambuf<char,std::char_traits<char> >"* %12, i32 0, i32 2		; <i8**>:13 [#uses=10]
	load i8** %13, align 4		; <i8*>:14 [#uses=2]
	getelementptr %"struct.std::basic_streambuf<char,std::char_traits<char> >"* %12, i32 0, i32 3		; <i8**>:15 [#uses=6]
	load i8** %15, align 4		; <i8*>:16 [#uses=1]
	icmp ult i8* %14, %16		; <i1>:17 [#uses=1]
	br i1 %17, label %bb81, label %bb82

bb81:		; preds = %bb
	load i8* %14, align 1		; <i8>:18 [#uses=1]
	zext i8 %18 to i32		; <i32>:19 [#uses=1]
	%.pre = getelementptr %"struct.std::basic_streambuf<char,std::char_traits<char> >"* %12, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	br label %bb119.preheader

bb82:		; preds = %bb
	getelementptr %"struct.std::basic_streambuf<char,std::char_traits<char> >"* %12, i32 0, i32 0		; <i32 (...)***>:20 [#uses=2]
	load i32 (...)*** %20, align 4		; <i32 (...)**>:21 [#uses=1]
	getelementptr i32 (...)** %21, i32 9		; <i32 (...)**>:22 [#uses=1]
	load i32 (...)** %22, align 4		; <i32 (...)*>:23 [#uses=1]
	bitcast i32 (...)* %23 to i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*		; <i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*>:24 [#uses=1]
	invoke i32 %24( %"struct.std::basic_streambuf<char,std::char_traits<char> >"* %12 )
			to label %bb119.preheader unwind label %lpad		; <i32>:25 [#uses=1]

bb119.preheader:		; preds = %bb82, %bb81
	%.pre-phi = phi i32 (...)*** [ %.pre, %bb81 ], [ %20, %bb82 ]		; <i32 (...)***> [#uses=4]
	%__c79.0.ph = phi i32 [ %19, %bb81 ], [ %25, %bb82 ]		; <i32> [#uses=1]
	sext i8 %__delim to i32		; <i32>:26 [#uses=1]
	br label %bb119

bb84:		; preds = %bb119
	sub i32 %__n, %82		; <i32>:27 [#uses=1]
	add i32 %27, -1		; <i32>:28 [#uses=2]
	load i8** %15, align 4		; <i8*>:29 [#uses=1]
	ptrtoint i8* %29 to i32		; <i32>:30 [#uses=1]
	load i8** %13, align 4		; <i8*>:31 [#uses=3]
	ptrtoint i8* %31 to i32		; <i32>:32 [#uses=2]
	sub i32 %30, %32		; <i32>:33 [#uses=2]
	icmp slt i32 %28, %33		; <i1>:34 [#uses=1]
	select i1 %34, i32 %28, i32 %33		; <i32>:35 [#uses=3]
	icmp sgt i32 %35, 1		; <i1>:36 [#uses=1]
	br i1 %36, label %bb90, label %bb99

bb90:		; preds = %bb84
	call i8* @memchr( i8* %31, i32 %26, i32 %35 ) nounwind readonly 		; <i8*>:37 [#uses=2]
	icmp eq i8* %37, null		; <i1>:38 [#uses=1]
	br i1 %38, label %bb93, label %bb92

bb92:		; preds = %bb90
	ptrtoint i8* %37 to i32		; <i32>:39 [#uses=1]
	sub i32 %39, %32		; <i32>:40 [#uses=1]
	br label %bb93

bb93:		; preds = %bb92, %bb90
	%__size.0 = phi i32 [ %40, %bb92 ], [ %35, %bb90 ]		; <i32> [#uses=4]
	call void @llvm.memcpy.i32( i8* %__s_addr.0, i8* %31, i32 %__size.0, i32 1 )
	getelementptr i8* %__s_addr.0, i32 %__size.0		; <i8*>:41 [#uses=3]
	load i8** %13, align 4		; <i8*>:42 [#uses=1]
	getelementptr i8* %42, i32 %__size.0		; <i8*>:43 [#uses=1]
	store i8* %43, i8** %13, align 4
	load i32* %0, align 4		; <i32>:44 [#uses=1]
	add i32 %44, %__size.0		; <i32>:45 [#uses=1]
	store i32 %45, i32* %0, align 4
	load i8** %13, align 4		; <i8*>:46 [#uses=2]
	load i8** %15, align 4		; <i8*>:47 [#uses=1]
	icmp ult i8* %46, %47		; <i1>:48 [#uses=1]
	br i1 %48, label %bb95, label %bb96

bb95:		; preds = %bb93
	load i8* %46, align 1		; <i8>:49 [#uses=1]
	zext i8 %49 to i32		; <i32>:50 [#uses=1]
	br label %bb119

bb96:		; preds = %bb93
	load i32 (...)*** %.pre-phi, align 4		; <i32 (...)**>:51 [#uses=1]
	getelementptr i32 (...)** %51, i32 9		; <i32 (...)**>:52 [#uses=1]
	load i32 (...)** %52, align 4		; <i32 (...)*>:53 [#uses=1]
	bitcast i32 (...)* %53 to i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*		; <i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*>:54 [#uses=1]
	invoke i32 %54( %"struct.std::basic_streambuf<char,std::char_traits<char> >"* %12 )
			to label %bb119 unwind label %lpad		; <i32>:55 [#uses=1]

bb99:		; preds = %bb84
	trunc i32 %__c79.0 to i8		; <i8>:56 [#uses=1]
	store i8 %56, i8* %__s_addr.0, align 1
	getelementptr i8* %__s_addr.0, i32 1		; <i8*>:57 [#uses=5]
	load i32* %0, align 4		; <i32>:58 [#uses=1]
	add i32 %58, 1		; <i32>:59 [#uses=1]
	store i32 %59, i32* %0, align 4
	load i8** %13, align 4		; <i8*>:60 [#uses=3]
	load i8** %15, align 4		; <i8*>:61 [#uses=1]
	icmp ult i8* %60, %61		; <i1>:62 [#uses=1]
	br i1 %62, label %bb101, label %bb102

bb101:		; preds = %bb99
	load i8* %60, align 1		; <i8>:63 [#uses=1]
	zext i8 %63 to i32		; <i32>:64 [#uses=1]
	getelementptr i8* %60, i32 1		; <i8*>:65 [#uses=1]
	store i8* %65, i8** %13, align 4
	br label %bb104

bb102:		; preds = %bb99
	load i32 (...)*** %.pre-phi, align 4		; <i32 (...)**>:66 [#uses=1]
	getelementptr i32 (...)** %66, i32 10		; <i32 (...)**>:67 [#uses=1]
	load i32 (...)** %67, align 4		; <i32 (...)*>:68 [#uses=1]
	bitcast i32 (...)* %68 to i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*		; <i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*>:69 [#uses=1]
	invoke i32 %69( %"struct.std::basic_streambuf<char,std::char_traits<char> >"* %12 )
			to label %bb104 unwind label %lpad		; <i32>:70 [#uses=1]

bb104:		; preds = %bb102, %bb101
	%__ret44.0 = phi i32 [ %64, %bb101 ], [ %70, %bb102 ]		; <i32> [#uses=1]
	icmp eq i32 %__ret44.0, -1		; <i1>:71 [#uses=1]
	br i1 %71, label %bb119, label %bb112

bb112:		; preds = %bb104
	load i8** %13, align 4		; <i8*>:72 [#uses=2]
	load i8** %15, align 4		; <i8*>:73 [#uses=1]
	icmp ult i8* %72, %73		; <i1>:74 [#uses=1]
	br i1 %74, label %bb114, label %bb115

bb114:		; preds = %bb112
	load i8* %72, align 1		; <i8>:75 [#uses=1]
	zext i8 %75 to i32		; <i32>:76 [#uses=1]
	br label %bb119

bb115:		; preds = %bb112
	load i32 (...)*** %.pre-phi, align 4		; <i32 (...)**>:77 [#uses=1]
	getelementptr i32 (...)** %77, i32 9		; <i32 (...)**>:78 [#uses=1]
	load i32 (...)** %78, align 4		; <i32 (...)*>:79 [#uses=1]
	bitcast i32 (...)* %79 to i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*		; <i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*>:80 [#uses=1]
	invoke i32 %80( %"struct.std::basic_streambuf<char,std::char_traits<char> >"* %12 )
			to label %bb119 unwind label %lpad		; <i32>:81 [#uses=1]

bb119:		; preds = %bb115, %bb114, %bb104, %bb96, %bb95, %bb119.preheader
	%__c79.0 = phi i32 [ %__c79.0.ph, %bb119.preheader ], [ %50, %bb95 ], [ %76, %bb114 ], [ %55, %bb96 ], [ -1, %bb104 ], [ %81, %bb115 ]		; <i32> [#uses=3]
	%__s_addr.0 = phi i8* [ %__s, %bb119.preheader ], [ %41, %bb95 ], [ %57, %bb114 ], [ %41, %bb96 ], [ %57, %bb104 ], [ %57, %bb115 ]		; <i8*> [#uses=5]
	load i32* %0, align 4		; <i32>:82 [#uses=2]
	add i32 %82, 1		; <i32>:83 [#uses=2]
	%.not = icmp sge i32 %83, %__n		; <i1> [#uses=1]
	icmp eq i32 %__c79.0, -1		; <i1>:84 [#uses=3]
	icmp eq i32 %__c79.0, %3		; <i1>:85 [#uses=2]
	%or.cond = or i1 %84, %85		; <i1> [#uses=1]
	%or.cond188 = or i1 %or.cond, %.not		; <i1> [#uses=1]
	br i1 %or.cond188, label %bb141, label %bb84

bb141:		; preds = %bb119
	%.not194 = xor i1 %85, true		; <i1> [#uses=1]
	%brmerge = or i1 %84, %.not194		; <i1> [#uses=1]
	%.mux = select i1 %84, i32 2, i32 4		; <i32> [#uses=0]
	br i1 %brmerge, label %bb162, label %bb146

bb146:		; preds = %bb141
	store i32 %83, i32* %0, align 4
	load i8** %13, align 4		; <i8*>:86 [#uses=2]
	load i8** %15, align 4		; <i8*>:87 [#uses=1]
	icmp ult i8* %86, %87		; <i1>:88 [#uses=1]
	br i1 %88, label %bb148, label %bb149

bb148:		; preds = %bb146
	getelementptr i8* %86, i32 1		; <i8*>:89 [#uses=1]
	store i8* %89, i8** %13, align 4
	ret %"struct.std::basic_istream<char,std::char_traits<char> >"* %this

bb149:		; preds = %bb146
	load i32 (...)*** %.pre-phi, align 4		; <i32 (...)**>:90 [#uses=1]
	getelementptr i32 (...)** %90, i32 10		; <i32 (...)**>:91 [#uses=1]
	load i32 (...)** %91, align 4		; <i32 (...)*>:92 [#uses=1]
	bitcast i32 (...)* %92 to i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*		; <i32 (%"struct.std::basic_streambuf<char,std::char_traits<char> >"*)*>:93 [#uses=1]
	invoke i32 %93( %"struct.std::basic_streambuf<char,std::char_traits<char> >"* %12 )
			to label %bb162 unwind label %lpad		; <i32>:94 [#uses=0]

bb162:		; preds = %bb149, %bb141, %entry
	ret %"struct.std::basic_istream<char,std::char_traits<char> >"* %this

lpad:		; preds = %bb149, %bb115, %bb102, %bb96, %bb82
	%__s_addr.1 = phi i8* [ %__s, %bb82 ], [ %__s_addr.0, %bb149 ], [ %41, %bb96 ], [ %57, %bb102 ], [ %57, %bb115 ]		; <i8*> [#uses=0]
	call void @__cxa_rethrow( ) noreturn 
	unreachable
}

declare i8* @__cxa_begin_catch(i8*) nounwind 

declare i8* @llvm.eh.exception() nounwind 

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...) nounwind 

declare void @__cxa_rethrow() noreturn 

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(...)

declare void @_ZNSi6sentryC1ERSib(%"struct.std::basic_istream<char,std::char_traits<char> >::sentry"*, %"struct.std::basic_istream<char,std::char_traits<char> >"*, i8 zeroext )

declare i8* @memchr(i8*, i32, i32) nounwind readonly 

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 

declare void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(%"struct.std::basic_ios<char,std::char_traits<char> >"*, i32)

declare extern_weak i32 @pthread_once(i32*, void ()*)

declare extern_weak i8* @pthread_getspecific(i32)

declare extern_weak i32 @pthread_setspecific(i32, i8*)

declare extern_weak i32 @pthread_create(i32*, %struct.pthread_attr_t*, i8* (i8*)*, i8*)

declare extern_weak i32 @pthread_cancel(i32)

declare extern_weak i32 @pthread_mutex_lock(%struct.pthread_mutex_t*)

declare extern_weak i32 @pthread_mutex_trylock(%struct.pthread_mutex_t*)

declare extern_weak i32 @pthread_mutex_unlock(%struct.pthread_mutex_t*)

declare extern_weak i32 @pthread_mutex_init(%struct.pthread_mutex_t*, %struct..0._11*)

declare extern_weak i32 @pthread_key_create(i32*, void (i8*)*)

declare extern_weak i32 @pthread_key_delete(i32)

declare extern_weak i32 @pthread_mutexattr_init(%struct..0._11*)

declare extern_weak i32 @pthread_mutexattr_settype(%struct..0._11*, i32)

declare extern_weak i32 @pthread_mutexattr_destroy(%struct..0._11*)
