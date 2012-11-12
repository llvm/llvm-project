; RUN: opt < %s -inline -prune-eh -disable-output
; PR992
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
deplibs = [ "stdc++", "c", "crtend" ]
	%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i32, [52 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
	%"struct.__cxxabiv1::__array_type_info" = type { %"struct.std::type_info" }
	%"struct.__cxxabiv1::__si_class_type_info" = type { %"struct.__cxxabiv1::__array_type_info", %"struct.__cxxabiv1::__array_type_info"* }
	%"struct.__gnu_cxx::_Rope_rep_alloc_base<char,std::allocator<char>, true>" = type { i32 }
	%"struct.__gnu_cxx::__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >" = type { i8* }
	%"struct.__gnu_cxx::__normal_iterator<const wchar_t*,std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> > >" = type { i32* }
	%"struct.__gnu_cxx::char_producer<char>" = type { i32 (...)** }
	%"struct.__gnu_cxx::stdio_sync_filebuf<char,std::char_traits<char> >" = type { %"struct.std::basic_streambuf<char,std::char_traits<char> >", %struct._IO_FILE*, i32 }
	%"struct.__gnu_cxx::stdio_sync_filebuf<wchar_t,std::char_traits<wchar_t> >" = type { %"struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >", %struct._IO_FILE*, i32 }
	%struct.__locale_struct = type { [13 x %struct.locale_data*], i16*, i32*, i32*, [13 x i8*] }
	%struct.__mbstate_t = type { i32, %"struct.__gnu_cxx::_Rope_rep_alloc_base<char,std::allocator<char>, true>" }
	%struct.locale_data = type opaque
	%"struct.std::__basic_file<char>" = type { %struct._IO_FILE*, i1 }
	%"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>" = type { %"struct.std::locale::facet" }
	%"struct.std::basic_filebuf<char,std::char_traits<char> >" = type { %"struct.std::basic_streambuf<char,std::char_traits<char> >", i32, %"struct.std::__basic_file<char>", i32, %struct.__mbstate_t, %struct.__mbstate_t, i8*, i32, i1, i1, i1, i1, i8, i8*, i8*, i1, %"struct.std::codecvt<char,char,__mbstate_t>"*, i8*, i32, i8*, i8* }
	%"struct.std::basic_filebuf<wchar_t,std::char_traits<wchar_t> >" = type { %"struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >", i32, %"struct.std::__basic_file<char>", i32, %struct.__mbstate_t, %struct.__mbstate_t, i32*, i32, i1, i1, i1, i1, i32, i32*, i32*, i1, %"struct.std::codecvt<char,char,__mbstate_t>"*, i8*, i32, i8*, i8* }
	%"struct.std::basic_fstream<char,std::char_traits<char> >" = type { { %"struct.std::locale::facet", %"struct.__gnu_cxx::char_producer<char>" }, %"struct.std::basic_filebuf<char,std::char_traits<char> >", %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_fstream<wchar_t,std::char_traits<wchar_t> >" = type { { %"struct.std::locale::facet", %"struct.__gnu_cxx::char_producer<char>" }, %"struct.std::basic_filebuf<wchar_t,std::char_traits<wchar_t> >", %"struct.std::basic_ios<wchar_t,std::char_traits<wchar_t> >" }
	%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8, i1, %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, %"struct.std::ctype<char>"*, %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"*, %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"* }
	%"struct.std::basic_ios<wchar_t,std::char_traits<wchar_t> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<wchar_t,std::char_traits<wchar_t> >"*, i32, i1, %"struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >"*, %"struct.std::codecvt<char,char,__mbstate_t>"*, %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"*, %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"* }
	%"struct.std::basic_iostream<wchar_t,std::char_traits<wchar_t> >" = type { %"struct.std::locale::facet", %"struct.__gnu_cxx::char_producer<char>", %"struct.std::basic_ios<wchar_t,std::char_traits<wchar_t> >" }
	%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { i32 (...)**, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_ostream<wchar_t,std::char_traits<wchar_t> >" = type { i32 (...)**, %"struct.std::basic_ios<wchar_t,std::char_traits<wchar_t> >" }
	%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"struct.std::locale" }
	%"struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >" = type { i32 (...)**, i32*, i32*, i32*, i32*, i32*, i32*, %"struct.std::locale" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" = type { %"struct.__gnu_cxx::__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" = type { i32, i32, i32 }
	%"struct.std::basic_string<wchar_t,std::char_traits<wchar_t>,std::allocator<wchar_t> >" = type { %"struct.__gnu_cxx::__normal_iterator<const wchar_t*,std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> > >" }
	%"struct.std::codecvt<char,char,__mbstate_t>" = type { %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>", %struct.__locale_struct* }
	%"struct.std::collate<char>" = type { %"struct.std::locale::facet", %struct.__locale_struct* }
	%"struct.std::collate_byname<char>" = type { %"struct.std::collate<char>" }
	%"struct.std::ctype<char>" = type { %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>", %struct.__locale_struct*, i1, i32*, i32*, i16* }
	%"struct.std::ctype_byname<char>" = type { %"struct.std::ctype<char>" }
	%"struct.std::domain_error" = type { %"struct.std::logic_error" }
	%"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"struct.std::locale" }
	%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
	%"struct.std::ios_base::_Words" = type { i8*, i32 }
	%"struct.std::istreambuf_iterator<char,std::char_traits<char> >" = type { %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, i32 }
	%"struct.std::istreambuf_iterator<wchar_t,std::char_traits<wchar_t> >" = type { %"struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >"*, i32 }
	%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
	%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
	%"struct.std::locale::facet" = type { i32 (...)**, i32 }
	%"struct.std::logic_error" = type { %"struct.__gnu_cxx::char_producer<char>", %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" }
	%"struct.std::type_info" = type { i32 (...)**, i8* }
@.str_11 = external global [42 x i8]		; <[42 x i8]*> [#uses=0]
@.str_9 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]
@.str_1 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]

define void @main() {
entry:
	tail call fastcc void @_ZNSolsEi( )
	ret void
}

define fastcc void @_ZNSolsEi() {
entry:
	%tmp.22 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp.22, label %else, label %then

then:		; preds = %entry
	ret void

else:		; preds = %entry
	tail call fastcc void @_ZNSolsEl( )
	ret void
}

define void @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_() {
entry:
	ret void
}

define fastcc void @_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate() {
entry:
	tail call fastcc void @_ZSt19__throw_ios_failurePKc( )
	ret void
}

define fastcc void @_ZNSo3putEc() {
entry:
	ret void
}

define fastcc void @_ZNSolsEl() {
entry:
	%tmp.21.i = icmp eq %"struct.std::basic_ostream<char,std::char_traits<char> >"* null, null		; <i1> [#uses=1]
	br i1 %tmp.21.i, label %endif.0.i, label %shortcirc_next.i

shortcirc_next.i:		; preds = %entry
	ret void

endif.0.i:		; preds = %entry
	call fastcc void @_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate( )
	ret void
}

define fastcc void @_ZSt19__throw_ios_failurePKc() {
entry:
	call fastcc void @_ZNSsC1EPKcRKSaIcE( )
	ret void
}

define fastcc void @_ZNSt8ios_baseD2Ev() {
entry:
	unreachable
}

define void @_ZN9__gnu_cxx18stdio_sync_filebufIwSt11char_traitsIwEE5uflowEv() {
entry:
	unreachable
}

define void @_ZN9__gnu_cxx18stdio_sync_filebufIcSt11char_traitsIcEED1Ev() {
entry:
	unreachable
}

define void @_ZNSt15basic_streambufIcSt11char_traitsIcEE6setbufEPci() {
entry:
	ret void
}

define fastcc void @_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale() {
entry:
	ret void
}

declare fastcc void @_ZNSaIcED1Ev()

define fastcc void @_ZSt19__throw_logic_errorPKc() {
entry:
	call fastcc void @_ZNSt11logic_errorC1ERKSs( )
	ret void
}

define fastcc void @_ZNSs4_Rep9_S_createEjRKSaIcE() {
entry:
	br i1 false, label %then.0, label %endif.0

then.0:		; preds = %entry
	call fastcc void @_ZSt20__throw_length_errorPKc( )
	ret void

endif.0:		; preds = %entry
	ret void
}

define fastcc void @_ZSt20__throw_length_errorPKc() {
entry:
	call fastcc void @_ZNSt12length_errorC1ERKSs( )
	ret void
}

define fastcc void @_ZNSs16_S_construct_auxIPKcEEPcT_S3_RKSaIcE12__false_type() {
entry:
	br i1 false, label %then.1.i, label %endif.1.i

then.1.i:		; preds = %entry
	call fastcc void @_ZSt19__throw_logic_errorPKc( )
	ret void

endif.1.i:		; preds = %entry
	call fastcc void @_ZNSs4_Rep9_S_createEjRKSaIcE( )
	unreachable
}

define fastcc void @_ZNSsC1ERKSs() {
entry:
	call fastcc void @_ZNSs4_Rep7_M_grabERKSaIcES2_( )
	invoke fastcc void @_ZNSaIcEC1ERKS_( )
			to label %invoke_cont.1 unwind label %invoke_catch.1

invoke_catch.1:		; preds = %entry
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 catch i8* null
	call fastcc void @_ZNSaIcED1Ev( )
	resume { i8*, i32 } %exn

invoke_cont.1:		; preds = %entry
	call fastcc void @_ZNSaIcEC2ERKS_( )
	ret void
}

define fastcc void @_ZNSs7reserveEj() {
entry:
	ret void
}

define fastcc void @_ZNSaIcEC1ERKS_() {
entry:
	ret void
}

define fastcc void @_ZNSs4_Rep7_M_grabERKSaIcES2_() {
entry:
	br i1 false, label %else.i, label %cond_true

cond_true:		; preds = %entry
	ret void

else.i:		; preds = %entry
	tail call fastcc void @_ZNSs4_Rep9_S_createEjRKSaIcE( )
	ret void
}

define fastcc void @_ZNSsC1EPKcRKSaIcE() {
entry:
	tail call fastcc void @_ZNSs16_S_construct_auxIPKcEEPcT_S3_RKSaIcE12__false_type( )
	unreachable
}

define fastcc void @_ZNSaIcEC2ERKS_() {
entry:
	ret void
}

define void @_ZNSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEED1Ev() {
entry:
	unreachable
}

define void @_ZNSt14collate_bynameIcED1Ev() {
entry:
	unreachable
}

define void @_ZNKSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE6do_getES3_S3_RSt8ios_baseRSt12_Ios_IostateRy() {
entry:
	ret void
}

define void @_ZNSt23__codecvt_abstract_baseIcc11__mbstate_tED1Ev() {
entry:
	unreachable
}

define void @_ZNSt12ctype_bynameIcED0Ev() {
entry:
	unreachable
}

define fastcc void @_ZNSt8messagesIwEC1Ej() {
entry:
	ret void
}

define fastcc void @_ZSt14__convert_to_vIlEvPKcRT_RSt12_Ios_IostateRKP15__locale_structi() {
entry:
	ret void
}

define fastcc void @_ZNSt8time_getIwSt19istreambuf_iteratorIwSt11char_traitsIwEEEC1Ej() {
entry:
	ret void
}

define fastcc void @_ZNSt8time_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEEC1Ej() {
entry:
	ret void
}

define fastcc void @_ZNKSt7num_getIwSt19istreambuf_iteratorIwSt11char_traitsIwEEE16_M_extract_floatES3_S3_RSt8ios_baseRSt12_Ios_IostateRSs() {
entry:
	unreachable
}

define fastcc void @_ZNSbIwSt11char_traitsIwESaIwEE4swapERS2_() {
entry:
	ret void
}

define void @_ZNSt14basic_iostreamIwSt11char_traitsIwEED0Ev() {
entry:
	unreachable
}

define void @_ZNSt15basic_streambufIcSt11char_traitsIcEE9showmanycEv() {
entry:
	ret void
}

define void @_ZNSt9exceptionD0Ev() {
entry:
	unreachable
}

define fastcc void @_ZNSt11logic_errorC1ERKSs() {
entry:
	call fastcc void @_ZNSsC1ERKSs( )
	ret void
}

define fastcc void @_ZNSt11logic_errorD2Ev() {
entry:
	unreachable
}

define fastcc void @_ZNSt12length_errorC1ERKSs() {
entry:
	invoke fastcc void @_ZNSsC1ERKSs( )
			to label %_ZNSt11logic_errorC2ERKSs.exit unwind label %invoke_catch.i

invoke_catch.i:		; preds = %entry
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 catch i8* null
	resume { i8*, i32 } %exn

_ZNSt11logic_errorC2ERKSs.exit:		; preds = %entry
	ret void
}

define void @_ZNK10__cxxabiv120__si_class_type_info20__do_find_public_srcEiPKvPKNS_17__class_type_infoES2_() {
entry:
	ret void
}

define fastcc void @_ZNSbIwSt11char_traitsIwESaIwEE16_S_construct_auxIPKwEEPwT_S7_RKS1_12__false_type() {
entry:
	ret void
}

define void @_ZTv0_n12_NSt13basic_fstreamIwSt11char_traitsIwEED1Ev() {
entry:
	ret void
}

define void @_ZNSt13basic_fstreamIcSt11char_traitsIcEED1Ev() {
entry:
	unreachable
}

define fastcc void @_ZNSt5ctypeIcEC1EPKtbj() {
entry:
	ret void
}

declare i32 @__gxx_personality_v0(...)
