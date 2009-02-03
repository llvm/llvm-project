; RUN: llvm-as < %s | opt -inline -prune-eh -disable-output
; PR993
target datalayout = "e-p:32:32"
target triple = "i386-unknown-openbsd3.9"
deplibs = [ "stdc++", "c", "crtend" ]
	%"struct.__gnu_cxx::__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >" = type { i8* }
	%"struct.__gnu_cxx::char_producer<char>" = type { i32 (...)** }
	%struct.__sFILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, i8*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__sbuf = type { i8*, i32 }
	%"struct.std::__basic_file<char>" = type { %struct.__sFILE*, i1 }
	%"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>" = type { %"struct.std::locale::facet" }
	%"struct.std::bad_alloc" = type { %"struct.__gnu_cxx::char_producer<char>" }
	%"struct.std::basic_filebuf<char,std::char_traits<char> >" = type { %"struct.std::basic_streambuf<char,std::char_traits<char> >", i32, %"struct.std::__basic_file<char>", i32, %union.__mbstate_t, %union.__mbstate_t, i8*, i32, i1, i1, i1, i1, i8, i8*, i8*, i1, %"struct.std::codecvt<char,char,__mbstate_t>"*, i8*, i32, i8*, i8* }
	%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8, i1, %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, %"struct.std::ctype<char>"*, %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"*, %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"* }
	%"struct.std::basic_iostream<char,std::char_traits<char> >" = type { %"struct.std::locale::facet", %"struct.__gnu_cxx::char_producer<char>", %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_ofstream<char,std::char_traits<char> >" = type { %"struct.__gnu_cxx::char_producer<char>", %"struct.std::basic_filebuf<char,std::char_traits<char> >", %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { i32 (...)**, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"struct.std::locale" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" = type { %"struct.__gnu_cxx::__normal_iterator<char*,std::basic_string<char, std::char_traits<char>, std::allocator<char> > >" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Rep_base" = type { i32, i32, i32 }
	%"struct.std::codecvt<char,char,__mbstate_t>" = type { %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>", i32* }
	%"struct.std::ctype<char>" = type { %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>", i32*, i1, i32*, i32*, i32* }
	%"struct.std::domain_error" = type { %"struct.std::logic_error" }
	%"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %struct.__sbuf, [8 x %struct.__sbuf], i32, %struct.__sbuf*, %"struct.std::locale" }
	%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
	%"struct.std::ios_base::_Words" = type { i8*, i32 }
	%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
	%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
	%"struct.std::locale::facet" = type { i32 (...)**, i32 }
	%"struct.std::logic_error" = type { %"struct.__gnu_cxx::char_producer<char>", %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" }
	%union.__mbstate_t = type { i64, [120 x i8] }
@.str_1 = external global [17 x i8]		; <[17 x i8]*> [#uses=0]
@.str_9 = external global [24 x i8]		; <[24 x i8]*> [#uses=0]

define void @main() {
entry:
	call fastcc void @_ZNSt14basic_ofstreamIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode( )
	ret void
}

define fastcc void @_ZNSt14basic_ofstreamIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode() {
entry:
	%tmp.6 = icmp eq %"struct.std::basic_filebuf<char,std::char_traits<char> >"* null, null		; <i1> [#uses=1]
	br i1 %tmp.6, label %then, label %UnifiedReturnBlock

then:		; preds = %entry
	tail call fastcc void @_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate( )
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define fastcc void @_ZN10__cxxabiv111__terminateEPFvvE() {
entry:
	unreachable
}

define void @_ZNSdD0Ev() {
entry:
	unreachable
}

define void @_ZThn8_NSdD1Ev() {
entry:
	ret void
}

define void @_ZNSt13basic_filebufIcSt11char_traitsIcEED0Ev() {
entry:
	ret void
}

define void @_ZNSt13basic_filebufIcSt11char_traitsIcEE9pbackfailEi() {
entry:
	unreachable
}

define fastcc void @_ZNSoD2Ev() {
entry:
	unreachable
}

define fastcc void @_ZNSt9basic_iosIcSt11char_traitsIcEED2Ev() {
entry:
	unreachable
}

define fastcc void @_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate() {
entry:
	tail call fastcc void @_ZSt19__throw_ios_failurePKc( )
	ret void
}

declare fastcc void @_ZNSaIcED1Ev()

define fastcc void @_ZNSsC1EPKcRKSaIcE() {
entry:
	tail call fastcc void @_ZNSs16_S_construct_auxIPKcEEPcT_S3_RKSaIcE12__false_type( )
	unreachable
}

define fastcc void @_ZSt14__convert_to_vIyEvPKcRT_RSt12_Ios_IostateRKPii() {
entry:
	ret void
}

define fastcc void @_ZNSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEEC1Ej() {
entry:
	ret void
}

define fastcc void @_ZSt19__throw_ios_failurePKc() {
entry:
	call fastcc void @_ZNSsC1EPKcRKSaIcE( )
	unwind
}

define void @_GLOBAL__D__ZSt23lexicographical_compareIPKaS1_EbT_S2_T0_S3_() {
entry:
	ret void
}

define void @_ZNSt9bad_allocD1Ev() {
entry:
	unreachable
}

define fastcc void @_ZSt19__throw_logic_errorPKc() {
entry:
	invoke fastcc void @_ZNSt11logic_errorC1ERKSs( )
			to label %try_exit.0 unwind label %try_catch.0

try_catch.0:		; preds = %entry
	unreachable

try_exit.0:		; preds = %entry
	unwind
}

define fastcc void @_ZNSt11logic_errorC1ERKSs() {
entry:
	call fastcc void @_ZNSsC1ERKSs( )
	ret void
}

define void @_ZNSt12domain_errorD1Ev() {
entry:
	unreachable
}

define fastcc void @_ZSt20__throw_length_errorPKc() {
entry:
	call fastcc void @_ZNSt12length_errorC1ERKSs( )
	unwind
}

define fastcc void @_ZNSt12length_errorC1ERKSs() {
entry:
	invoke fastcc void @_ZNSsC1ERKSs( )
			to label %_ZNSt11logic_errorC2ERKSs.exit unwind label %invoke_catch.i

invoke_catch.i:		; preds = %entry
	unwind

_ZNSt11logic_errorC2ERKSs.exit:		; preds = %entry
	ret void
}

define fastcc void @_ZNSs4_Rep9_S_createEjRKSaIcE() {
entry:
	call fastcc void @_ZSt20__throw_length_errorPKc( )
	unreachable
}

define fastcc void @_ZNSs12_S_constructIN9__gnu_cxx17__normal_iteratorIPcSsEEEES2_T_S4_RKSaIcESt20forward_iterator_tag() {
entry:
	unreachable
}

define fastcc void @_ZNSs16_S_construct_auxIPKcEEPcT_S3_RKSaIcE12__false_type() {
entry:
	br i1 false, label %then.1.i, label %endif.1.i

then.1.i:		; preds = %entry
	call fastcc void @_ZSt19__throw_logic_errorPKc( )
	br label %endif.1.i

endif.1.i:		; preds = %then.1.i, %entry
	call fastcc void @_ZNSs4_Rep9_S_createEjRKSaIcE( )
	unreachable
}

define fastcc void @_ZNSsC1ERKSs() {
entry:
	call fastcc void @_ZNSs4_Rep7_M_grabERKSaIcES2_( )
	invoke fastcc void @_ZNSaIcEC1ERKS_( )
			to label %invoke_cont.1 unwind label %invoke_catch.1

invoke_catch.1:		; preds = %entry
	call fastcc void @_ZNSaIcED1Ev( )
	unwind

invoke_cont.1:		; preds = %entry
	call fastcc void @_ZNSaIcEC2ERKS_( )
	ret void
}

define fastcc void @_ZNSaIcEC1ERKS_() {
entry:
	ret void
}

define fastcc void @_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_jc() {
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
	unreachable
}

define fastcc void @_ZNSaIcEC2ERKS_() {
entry:
	ret void
}

define fastcc void @_ZN9__gnu_cxx12__pool_allocILb1ELi0EE8allocateEj() {
entry:
	ret void
}

define fastcc void @_ZN9__gnu_cxx12__pool_allocILb1ELi0EE9_S_refillEj() {
entry:
	unreachable
}
