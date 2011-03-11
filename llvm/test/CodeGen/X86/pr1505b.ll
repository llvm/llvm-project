; RUN: llc < %s -mcpu=i486 | grep fstpl | count 5
; RUN: llc < %s -mcpu=i486 | grep fstps | count 2
; PR1505

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
	%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8, i8, %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, %"struct.std::ctype<char>"*, %"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >"*, %"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >"* }
	%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { i32 (...)**, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"struct.std::locale" }
	%"struct.std::ctype<char>" = type { %"struct.std::locale::facet", i32*, i8, i32*, i32*, i32*, i8, [256 x i8], [256 x i8], i8 }
	%"struct.std::ctype_base" = type <{ i8 }>
	%"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"struct.std::locale" }
	%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
	%"struct.std::ios_base::_Words" = type { i8*, i32 }
	%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
	%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
	%"struct.std::locale::facet" = type { i32 (...)**, i32 }
	%"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >" = type { %"struct.std::locale::facet" }
@a = global float 0x3FD3333340000000		; <float*> [#uses=1]
@b = global double 6.000000e-01, align 8		; <double*> [#uses=1]
@_ZSt8__ioinit = internal global %"struct.std::ctype_base" zeroinitializer		; <%"struct.std::ctype_base"*> [#uses=2]
@__dso_handle = external global i8*		; <i8**> [#uses=1]
@_ZSt4cout = external global %"struct.std::basic_ostream<char,std::char_traits<char> >"		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=2]
@.str = internal constant [12 x i8] c"tan float: \00"		; <[12 x i8]*> [#uses=1]
@.str1 = internal constant [13 x i8] c"tan double: \00"		; <[13 x i8]*> [#uses=1]

declare void @_ZNSt8ios_base4InitD1Ev(%"struct.std::ctype_base"*)

declare void @_ZNSt8ios_base4InitC1Ev(%"struct.std::ctype_base"*)

declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)

define i32 @main() {
entry:
	%tmp6 = volatile load float* @a		; <float> [#uses=1]
	%tmp9 = tail call float @tanf( float %tmp6 )		; <float> [#uses=1]
	%tmp12 = volatile load double* @b		; <double> [#uses=1]
	%tmp13 = tail call double @tan( double %tmp12 )		; <double> [#uses=1]
	%tmp1314 = fptrunc double %tmp13 to float		; <float> [#uses=1]
	%tmp16 = tail call %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc( %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZSt4cout, i8* getelementptr ([12 x i8]* @.str, i32 0, i32 0) )		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=1]
	%tmp1920 = fpext float %tmp9 to double		; <double> [#uses=1]
	%tmp22 = tail call %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSolsEd( %"struct.std::basic_ostream<char,std::char_traits<char> >"* %tmp16, double %tmp1920 )		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=1]
	%tmp30 = tail call %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_( %"struct.std::basic_ostream<char,std::char_traits<char> >"* %tmp22 )		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=0]
	%tmp34 = tail call %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc( %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZSt4cout, i8* getelementptr ([13 x i8]* @.str1, i32 0, i32 0) )		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=1]
	%tmp3940 = fpext float %tmp1314 to double		; <double> [#uses=1]
	%tmp42 = tail call %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSolsEd( %"struct.std::basic_ostream<char,std::char_traits<char> >"* %tmp34, double %tmp3940 )		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=1]
	%tmp51 = tail call %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_( %"struct.std::basic_ostream<char,std::char_traits<char> >"* %tmp42 )		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=0]
	ret i32 0
}

declare float @tanf(float)

declare double @tan(double)

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8*)

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSolsEd(%"struct.std::basic_ostream<char,std::char_traits<char> >"*, double)

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"struct.std::basic_ostream<char,std::char_traits<char> >"*)
