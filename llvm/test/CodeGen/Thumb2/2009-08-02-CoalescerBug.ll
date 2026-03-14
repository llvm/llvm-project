; RUN: llc < %s -mtriple=thumbv7-apple-darwin9 -mcpu=cortex-a8 -relocation-model=pic -frame-pointer=all

	%0 = type { ptr, i32 }		; type %0
	%1 = type { ptr, i32 }		; type %1
	%2 = type { ptr, i32 }		; type %2
	%3 = type { ptr, i32 }		; type %3
	%4 = type { ptr, i32 }		; type %4
	%"struct.std::CharVectorType" = type { %"struct.std::_Vector_base<char,std::allocator<char> >" }
	%"struct.std::_Bit_const_iterator" = type { %"struct.std::_Bit_iterator_base" }
	%"struct.std::_Bit_iterator_base" = type { ptr, i32 }
	%"struct.std::_Bvector_base<std::allocator<bool> >" = type { %"struct.std::_Bvector_base<std::allocator<bool> >::_Bvector_impl" }
	%"struct.std::_Bvector_base<std::allocator<bool> >::_Bvector_impl" = type { %"struct.std::_Bit_const_iterator", %"struct.std::_Bit_const_iterator", ptr }
	%"struct.std::_Vector_base<char,std::allocator<char> >" = type { %"struct.std::_Vector_base<char,std::allocator<char> >::_Vector_impl" }
	%"struct.std::_Vector_base<char,std::allocator<char> >::_Vector_impl" = type { ptr, ptr, ptr }
	%"struct.std::_Vector_base<short unsigned int,std::allocator<short unsigned int> >" = type { %"struct.std::_Vector_base<short unsigned int,std::allocator<short unsigned int> >::_Vector_impl" }
	%"struct.std::_Vector_base<short unsigned int,std::allocator<short unsigned int> >::_Vector_impl" = type { ptr, ptr, ptr }
	%"struct.std::basic_ostream<char,std::char_traits<char> >.base" = type { ptr }
	%"struct.std::vector<bool,std::allocator<bool> >" = type { %"struct.std::_Bvector_base<std::allocator<bool> >" }
	%"struct.std::vector<short unsigned int,std::allocator<short unsigned int> >" = type { %"struct.std::_Vector_base<short unsigned int,std::allocator<short unsigned int> >" }
	%"struct.xalanc_1_8::FormatterListener" = type { %"struct.std::basic_ostream<char,std::char_traits<char> >.base", ptr, i32 }
	%"struct.xalanc_1_8::FormatterToXML" = type { %"struct.xalanc_1_8::FormatterListener", ptr, ptr, i16, [256 x i16], [256 x i16], i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, %"struct.xalanc_1_8::XalanDOMString", %"struct.xalanc_1_8::XalanDOMString", %"struct.xalanc_1_8::XalanDOMString", i32, i32, %"struct.std::vector<bool,std::allocator<bool> >", %"struct.xalanc_1_8::XalanDOMString", i8, i8, i8, i8, i8, %"struct.xalanc_1_8::XalanDOMString", %"struct.xalanc_1_8::XalanDOMString", %"struct.xalanc_1_8::XalanDOMString", %"struct.xalanc_1_8::XalanDOMString", %"struct.std::vector<short unsigned int,std::allocator<short unsigned int> >", i32, %"struct.std::CharVectorType", %"struct.std::vector<bool,std::allocator<bool> >", %0, %1, %2, %3, %0, %1, %2, %3, %4, ptr, i32 }
	%"struct.xalanc_1_8::XalanDOMString" = type { %"struct.std::vector<short unsigned int,std::allocator<short unsigned int> >", i32 }
	%"struct.xalanc_1_8::XalanOutputStream" = type { ptr, i32, ptr, i32, %"struct.std::vector<short unsigned int,std::allocator<short unsigned int> >", %"struct.xalanc_1_8::XalanDOMString", i8, i8, %"struct.std::CharVectorType" }

declare void @_ZN10xalanc_1_814FormatterToXML17writeParentTagEndEv(ptr)

define void @_ZN10xalanc_1_814FormatterToXML5cdataEPKtj(ptr %this, ptr %ch, i32 %length) {
entry:
	%0 = getelementptr %"struct.xalanc_1_8::FormatterToXML", ptr %this, i32 0, i32 13		; <ptr> [#uses=1]
	br i1 undef, label %bb4, label %bb

bb:		; preds = %entry
	store i8 0, ptr %0, align 1
	%1 = getelementptr %"struct.xalanc_1_8::FormatterToXML", ptr %this, i32 0, i32 0, i32 0, i32 0		; <ptr> [#uses=1]
	%2 = load ptr, ptr %1, align 4		; <ptr> [#uses=1]
	%3 = getelementptr ptr, ptr %2, i32 11		; <ptr> [#uses=1]
	%4 = load ptr, ptr %3, align 4		; <ptr> [#uses=1]
	tail call  void %4(ptr %this, ptr %ch, i32 %length)
	ret void

bb4:		; preds = %entry
	tail call  void @_ZN10xalanc_1_814FormatterToXML17writeParentTagEndEv(ptr %this)
	tail call  void undef(ptr %this, ptr %ch, i32 0, i32 %length, i8 zeroext undef)
	ret void
}
