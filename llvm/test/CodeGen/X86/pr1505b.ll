; RUN: llc < %s -mcpu=i486 | FileCheck %s
; PR1505

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
	%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
	%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { ptr, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, %"struct.std::locale" }
	%"struct.std::ctype<char>" = type { %"struct.std::locale::facet", ptr, i8, ptr, ptr, ptr, i8, [256 x i8], [256 x i8], i8 }
	%"struct.std::ctype_base" = type <{ i8 }>
	%"struct.std::ios_base" = type { ptr, i32, i32, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"struct.std::locale" }
	%"struct.std::ios_base::_Callback_list" = type { ptr, ptr, i32, i32 }
	%"struct.std::ios_base::_Words" = type { ptr, i32 }
	%"struct.std::locale" = type { ptr }
	%"struct.std::locale::_Impl" = type { i32, ptr, i32, ptr, ptr }
	%"struct.std::locale::facet" = type { ptr, i32 }
	%"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >" = type { %"struct.std::locale::facet" }
@a = global float 0x3FD3333340000000		; <ptr> [#uses=1]
@b = global double 6.000000e-01, align 8		; <ptr> [#uses=1]
@_ZSt8__ioinit = internal global %"struct.std::ctype_base" zeroinitializer		; <ptr> [#uses=2]
@__dso_handle = external global ptr		; <ptr> [#uses=1]
@_ZSt4cout = external global %"struct.std::basic_ostream<char,std::char_traits<char> >"		; <ptr> [#uses=2]
@.str = internal constant [12 x i8] c"tan float: \00"		; <ptr> [#uses=1]
@.str1 = internal constant [13 x i8] c"tan double: \00"		; <ptr> [#uses=1]

declare void @_ZNSt8ios_base4InitD1Ev(ptr)

declare void @_ZNSt8ios_base4InitC1Ev(ptr)

declare i32 @__cxa_atexit(ptr, ptr, ptr)

; CHECK: main
define i32 @main() {
entry:
; CHECK: flds
	%tmp6 = load volatile float, ptr @a		; <float> [#uses=1]
; CHECK: fstps (%esp)
; CHECK: tanf
	%tmp9 = tail call float @tanf( float %tmp6 )		; <float> [#uses=1]
; Spill returned value:
; CHECK: fstp

; CHECK: fldl
	%tmp12 = load volatile double, ptr @b		; <double> [#uses=1]
; CHECK: fstpl (%esp)
; CHECK: tan
	%tmp13 = tail call double @tan( double %tmp12 )		; <double> [#uses=1]
; Spill returned value:
; CHECK: fstp
	%tmp1314 = fptrunc double %tmp13 to float		; <float> [#uses=1]
	%tmp16 = tail call ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc( ptr @_ZSt4cout, ptr @.str )		; <ptr> [#uses=1]
	%tmp1920 = fpext float %tmp9 to double		; <double> [#uses=1]
; reload:
; CHECK: fld
; CHECK: fstpl
; CHECK: ZNSolsEd
	%tmp22 = tail call ptr @_ZNSolsEd( ptr %tmp16, double %tmp1920 )		; <ptr> [#uses=1]
	%tmp30 = tail call ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_( ptr %tmp22 )		; <ptr> [#uses=0]
; reload:
; CHECK: ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	%tmp34 = tail call ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc( ptr @_ZSt4cout, ptr @.str1 )		; <ptr> [#uses=1]
	%tmp3940 = fpext float %tmp1314 to double		; <double> [#uses=1]
; CHECK: fld
; CHECK: fstpl
; CHECK: ZNSolsEd
	%tmp42 = tail call ptr @_ZNSolsEd( ptr %tmp34, double %tmp3940 )		; <ptr> [#uses=1]
	%tmp51 = tail call ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_( ptr %tmp42 )		; <ptr> [#uses=0]
	ret i32 0
}

declare float @tanf(float)

declare double @tan(double)

declare ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr, ptr)

declare ptr @_ZNSolsEd(ptr, double)

declare ptr @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(ptr)
