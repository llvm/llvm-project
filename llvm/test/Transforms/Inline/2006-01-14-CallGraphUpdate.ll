; RUN: opt < %s -inline -prune-eh -disable-output

        %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>" = type { %"struct.std::locale::facet" }
        %"struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >" = type { i32 (...)**, i32*, i32*, i32*, i32*, i32*, i32*, %"struct.std::locale" }
        %"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"struct.std::locale" }
        %"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
        %"struct.std::ios_base::_Words" = type { i8*, i32 }
        %"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
        %"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
        %"struct.std::locale::facet" = type { i32 (...)**, i32 }
        %"struct.std::ostreambuf_iterator<wchar_t,std::char_traits<wchar_t> >" = type { %"struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >"*, i32 }

define void @_ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE6do_putES3_RSt8ios_basewl(%"struct.std::ostreambuf_iterator<wchar_t,std::char_traits<wchar_t> >"* %agg.result, %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"* %this, %"struct.std::basic_streambuf<wchar_t,std::char_traits<wchar_t> >"* %__s.0__, i32 %__s.1__, %"struct.std::ios_base"* %__io, i32 %__fill, i32 %__v) {
entry:
        tail call fastcc void @_ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE13_M_insert_intIlEES3_S3_RSt8ios_basewT_( )
        ret void
}

define fastcc void @_ZNKSt7num_putIwSt19ostreambuf_iteratorIwSt11char_traitsIwEEE13_M_insert_intIlEES3_S3_RSt8ios_basewT_() {
entry:
        %tmp.38 = shl i32 0, 3          ; <i32> [#uses=1]
        %tmp.39 = alloca i8, i32 %tmp.38                ; <i8*> [#uses=0]
        ret void
}

