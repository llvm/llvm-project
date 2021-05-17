; RUN: llc < %s
; PR5495

%"struct.std::__ctype_abstract_base<wchar_t>" = type { %"struct.std::locale::facet" }
%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8, i8, %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, %"struct.std::ctype<char>"*, %"struct.std::__ctype_abstract_base<wchar_t>"*, %"struct.std::__ctype_abstract_base<wchar_t>"* }
%"struct.std::basic_istream<char,std::char_traits<char> >" = type { i32 (...)**, i32, %"struct.std::basic_ios<char,std::char_traits<char> >" }
%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { i32 (...)**, %"struct.std::basic_ios<char,std::char_traits<char> >" }
%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"struct.std::locale" }
%"struct.std::ctype<char>" = type { %"struct.std::locale::facet", i32*, i8, i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8 }
%"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"struct.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i32 }
%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
%"struct.std::locale::facet" = type { i32 (...)**, i32 }
%union..0._15 = type { i32 }

declare i8* @__cxa_begin_catch(i8*) nounwind

declare %"struct.std::ctype<char>"* @_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale(%"struct.std::locale"*)

define %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_(%"struct.std::basic_istream<char,std::char_traits<char> >"* %__in, i8* nocapture %__s) personality i32 (...)* @__gxx_personality_v0 {
entry:
  %0 = invoke %"struct.std::ctype<char>"* @_ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale(%"struct.std::locale"* undef)
          to label %invcont8 unwind label %lpad74 ; <%"struct.std::ctype<char>"*> [#uses=0]

invcont8:                                         ; preds = %entry
  %1 = invoke i32 undef(%"struct.std::basic_streambuf<char,std::char_traits<char> >"* undef)
          to label %bb26.preheader unwind label %lpad ; <i32> [#uses=0]

bb26.preheader:                                   ; preds = %invcont8
  br label %invcont38

bb1.i100:                                         ; preds = %invcont38
  %2 = add nsw i32 1, %__extracted.0  ; <i32> [#uses=3]
  br i1 undef, label %bb.i97, label %bb1.i

bb.i97:                                           ; preds = %bb1.i100
  br label %invcont38

bb1.i:                                            ; preds = %bb1.i100
  %3 = invoke i32 undef(%"struct.std::basic_streambuf<char,std::char_traits<char> >"* undef)
          to label %invcont38 unwind label %lpad ; <i32> [#uses=0]

invcont24:                                        ; preds = %invcont38
  %4 = invoke i32 undef(%"struct.std::basic_streambuf<char,std::char_traits<char> >"* undef)
          to label %_ZNSt15basic_streambufIcSt11char_traitsIcEE6sbumpcEv.exit.i unwind label %lpad ; <i32> [#uses=0]

_ZNSt15basic_streambufIcSt11char_traitsIcEE6sbumpcEv.exit.i: ; preds = %invcont24
  br i1 undef, label %invcont25, label %bb.i93

bb.i93:                                           ; preds = %_ZNSt15basic_streambufIcSt11char_traitsIcEE6sbumpcEv.exit.i
  %5 = invoke i32 undef(%"struct.std::basic_streambuf<char,std::char_traits<char> >"* undef)
          to label %invcont25 unwind label %lpad ; <i32> [#uses=0]

invcont25:                                        ; preds = %bb.i93, %_ZNSt15basic_streambufIcSt11char_traitsIcEE6sbumpcEv.exit.i
  br label %invcont38

invcont38:                                        ; preds = %invcont25, %bb1.i, %bb.i97, %bb26.preheader
  %__extracted.0 = phi i32 [ 0, %bb26.preheader ], [ undef, %invcont25 ], [ %2, %bb.i97 ], [ %2, %bb1.i ] ; <i32> [#uses=1]
  br i1 false, label %bb1.i100, label %invcont24

lpad:                                             ; preds = %bb.i93, %invcont24, %bb1.i, %invcont8
  %__extracted.1 = phi i32 [ 0, %invcont8 ], [ %2, %bb1.i ], [ undef, %bb.i93 ], [ undef, %invcont24 ] ; <i32> [#uses=0]
  %lpad1 = landingpad { i8*, i32 }
            catch i8* null
  %eh_ptr = extractvalue { i8*, i32 } %lpad1, 0
  %6 = call i8* @__cxa_begin_catch(i8* %eh_ptr) nounwind ; <i8*> [#uses=0]
  unreachable

lpad74:                                           ; preds = %entry
  %lpad2 = landingpad { i8*, i32 }
            cleanup
  unreachable
}

declare i32 @__gxx_personality_v0(...) addrspace(0)
