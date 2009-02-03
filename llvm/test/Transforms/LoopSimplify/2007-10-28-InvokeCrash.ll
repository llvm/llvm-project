; RUN: llvm-as < %s | opt -loopsimplify -disable-output
; PR1752
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-s0:0:64-f80:32:32"
target triple = "i686-pc-mingw32"
	%struct.BigInt = type { %"struct.std::vector<ulong,std::allocator<ulong> >" }
	%struct.Fibonacci = type { %"struct.std::vector<BigInt,std::allocator<BigInt> >" }
	%struct.__false_type = type <{ i8 }>
	%"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >" = type { %struct.BigInt* }
	%"struct.std::_Vector_base<BigInt,std::allocator<BigInt> >" = type { %"struct.std::_Vector_base<BigInt,std::allocator<BigInt> >::_Vector_impl" }
	%"struct.std::_Vector_base<BigInt,std::allocator<BigInt> >::_Vector_impl" = type { %struct.BigInt*, %struct.BigInt*, %struct.BigInt* }
	%"struct.std::_Vector_base<ulong,std::allocator<ulong> >" = type { %"struct.std::_Vector_base<ulong,std::allocator<ulong> >::_Vector_impl" }
	%"struct.std::_Vector_base<ulong,std::allocator<ulong> >::_Vector_impl" = type { i32*, i32*, i32* }
	%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8, i8, %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, %"struct.std::ctype<char>"*, %"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >"*, %"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >"* }
	%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { i32 (...)**, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"struct.std::locale" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" = type { %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" }
	%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >::_Alloc_hider" = type { i8* }
	%"struct.std::basic_stringbuf<char,std::char_traits<char>,std::allocator<char> >" = type { %"struct.std::basic_streambuf<char,std::char_traits<char> >", i32, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >" }
	%"struct.std::ctype<char>" = type { %"struct.std::locale::facet", i32*, i8, i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8 }
	%"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"struct.std::locale" }
	%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
	%"struct.std::ios_base::_Words" = type { i8*, i32 }
	%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
	%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
	%"struct.std::locale::facet" = type { i32 (...)**, i32 }
	%"struct.std::num_get<char,std::istreambuf_iterator<char, std::char_traits<char> > >" = type { %"struct.std::locale::facet" }
	%"struct.std::ostringstream" = type { [4 x i8], %"struct.std::basic_stringbuf<char,std::char_traits<char>,std::allocator<char> >", %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::vector<BigInt,std::allocator<BigInt> >" = type { %"struct.std::_Vector_base<BigInt,std::allocator<BigInt> >" }
	%"struct.std::vector<ulong,std::allocator<ulong> >" = type { %"struct.std::_Vector_base<ulong,std::allocator<ulong> >" }
@.str13 = external constant [6 x i8]		; <[6 x i8]*> [#uses=1]
@.str14 = external constant [5 x i8]		; <[5 x i8]*> [#uses=1]
@.str15 = external constant [2 x i8]		; <[2 x i8]*> [#uses=1]
@_ZSt4cout = external global %"struct.std::basic_ostream<char,std::char_traits<char> >"		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=1]

declare void @_ZN9Fibonacci10get_numberEj(%struct.BigInt* sret , %struct.Fibonacci*, i32)

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8*)

declare void @_ZNSsD1Ev(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*)

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSolsEm(%"struct.std::basic_ostream<char,std::char_traits<char> >"*, i32)

declare void @_ZNKSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* sret , %"struct.std::ostringstream"*)

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(%"struct.std::basic_ostream<char,std::char_traits<char> >"*, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"*)

declare void @_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"struct.std::ostringstream"*)

declare %"struct.std::basic_ostream<char,std::char_traits<char> >"* @___ZlsRSoRK6BigInt___ZN9__gnu_cxx13new_allocatorI6BigIntE10deallocateEPS1_j(i32, %"struct.std::basic_ostream<char,std::char_traits<char> >"*, %struct.BigInt*, %struct.__false_type*, i32)

declare void @___ZNSt12_Vector_baseI6BigIntSaIS0_EE13_M_deallocateEPS0_j___ZNSt12_Vector_baseI6BigIntSaIS0_EED2Ev___ZNSt6vectorI6BigIntSaIS0_EEC1ERKS1_(%"struct.std::_Vector_base<BigInt,std::allocator<BigInt> >"*, i32, %struct.BigInt*, i32, %"struct.std::vector<BigInt,std::allocator<BigInt> >"*, %struct.__false_type*)

declare i32 @___ZN9__gnu_cxxmiIPK6BigIntS3_St6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T1_E15difference_typeERKSA_RKNS7_IT0_S9_EE___ZNKSt6vectorI6BigIntSaIS0_EE4sizeEv___ZNK9Fibonacci16show_all_numbersEv___ZNKSt6vectorI6BigIntSaIS0_EE8capacityEv(%"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"*, %"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"*, %"struct.std::vector<BigInt,std::allocator<BigInt> >"*, i32, %struct.Fibonacci*)

declare %struct.BigInt* @___ZNSt6vectorI6BigIntSaIS0_EEixEj___ZNSt6vectorI6BigIntSaIS0_EE3endEv(%"struct.std::vector<BigInt,std::allocator<BigInt> >"*, i32, i32)

declare %"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"* @___ZN9__gnu_cxx17__normal_iteratorIP6BigIntSt6vectorIS1_SaIS1_EEEppEv___ZNSt6vectorImSaImEED1Ev___ZN6BigIntD1Ev___ZN9__gnu_cxx13new_allocatorI6BigIntE7destroyEPS1____ZSt8_DestroyIP6BigIntSaIS0_EEvT_S3_T0_(i32, %"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"*, %"struct.std::vector<ulong,std::allocator<ulong> >"*, %struct.BigInt*, %struct.__false_type*, %struct.BigInt*, %struct.__false_type* noalias )

declare void @___ZNSt6vectorI6BigIntSaIS0_EED1Ev___ZN9FibonacciD1Ev___ZNSt6vectorImSaImEEC1ERKS0_(i32, %"struct.std::vector<BigInt,std::allocator<BigInt> >"*, %struct.Fibonacci*, %"struct.std::vector<ulong,std::allocator<ulong> >"*, %struct.__false_type*)

define void @___ZN9FibonacciC1Ej___ZN9Fibonacci11show_numberEm(%struct.Fibonacci* %this_this, i32 %functionID, i32 %n_i_n_i) {
bb_init:
	br label %bb_main

bb_main:		; preds = %meshBB349, %meshBB348, %meshBB347, %meshBB346, %meshBB345.unwinddest, %meshBB345, %meshBB344, %meshBB343, %meshBB342, %meshBB341, %meshBB340.normaldest, %meshBB340, %meshBB339, %invcont17.normaldest.normaldest, %invcont17.normaldest, %meshBB338.unwinddest, %meshBB338, %meshBB337.unwinddest, %meshBB337, %meshBB336.unwinddest, %meshBB336, %meshBB335, %meshBB334, %meshBB333, %meshBB332, %meshBB331, %meshBB330.normaldest, %meshBB330, %meshBB329.normaldest, %meshBB329, %meshBB328, %meshBB327, %meshBB326, %meshBB325.unwinddest, %meshBB325, %meshBB324, %meshBB323.normaldest, %meshBB323, %meshBB322.unwinddest, %meshBB322, %meshBB321, %meshBB320.unwinddest, %meshBB320, %meshBB319.unwinddest, %meshBB319, %meshBB318.unwinddest, %meshBB318, %meshBB317, %meshBB37.fragment, %meshBB37.unwinddest, %meshBB37, %meshBB36.fragment, %meshBB36, %meshBB35.fragment, %meshBB35, %meshBB34.fragment, %meshBB34, %meshBB33.fragment, %meshBB33, %meshBB32.fragment, %meshBB32, %meshBB31.fragment, %meshBB31, %meshBB30.fragment, %meshBB30.normaldest, %meshBB30, %meshBB29.fragment, %meshBB29.unwinddest, %meshBB29, %meshBB28.fragment, %meshBB28.unwinddest, %meshBB28, %meshBB27.fragment, %meshBB27, %meshBB26.fragment, %meshBB26.normaldest, %meshBB26, %meshBB25.fragment, %meshBB25, %meshBB24.fragment, %meshBB24.unwinddest, %meshBB24, %meshBB23.fragment, %meshBB23.normaldest, %meshBB23, %entry1.fragment.normaldest.normaldest, %entry1.fragment.normaldest, %meshBB22.fragment, %meshBB22.unwinddest, %meshBB22, %meshBB.fragment, %meshBB.unwinddest, %meshBB, %Unwind20, %unwind78.Unwind_crit_edge, %unwind78.fragment.fragment, %unwind78.fragment, %unwind78.fragment316, %unwind78, %invcont70, %unwind66.Unwind_crit_edge, %unwind66.fragment.fragment, %unwind66.fragment, %unwind66.fragment315, %unwind66, %unwind53.nofilter_crit_edge, %unwind53.fragment.fragment, %unwind53.fragment, %unwind53.fragment314, %unwind53, %nofilter.Unwind_crit_edge.normaldest, %nofilter.Unwind_crit_edge, %nofilter, %unwind43.nofilter_crit_edge, %unwind43.fragment.fragment, %unwind43.fragment, %unwind43.fragment313, %unwind43, %invcont41.normaldest, %invcont41, %unwind37.nofilter_crit_edge, %unwind37, %invcont36, %invcont33.unwind_crit_edge.unwinddest, %invcont33.unwind_crit_edge, %invcont30.unwind_crit_edge.unwinddest, %invcont30.unwind_crit_edge, %invcont30.normaldest, %invcont30, %invcont28.unwind_crit_edge, %invcont28.normaldest, %invcont28, %invcont25.unwind_crit_edge.unwinddest, %invcont25.unwind_crit_edge, %invcont25, %invcont22.unwind_crit_edge, %invcont22, %invcont17.unwind_crit_edge, %invcont17, %cond_next.unwind_crit_edge, %cond_next, %invcont12.cond_next_crit_edge, %invcont12.unwind_crit_edge, %invcont12, %cond_true.unwind_crit_edge.unwinddest, %cond_true.unwind_crit_edge, %invcont.cond_next_crit_edge, %invcont16.fragment, %invcont16, %unwind11.fragment, %unwind11, %entry.unwind_crit_edge, %entry1.fragment, %entry1.fragment312, %entry1, %Unwind, %unwind20.Unwind_crit_edge, %unwind20.fragment.fragment, %unwind20.fragment, %unwind20.fragment311, %unwind20, %invcont15, %invcont14.unwind10_crit_edge, %invcont14, %unwind10.Unwind_crit_edge, %unwind10.fragment, %unwind10.fragment310, %unwind10, %invcont.unwind10_crit_edge, %invcont, %unwind.fragment, %unwind, %entry.fragment, %entry.fragment309, %entry, %NewDefault, %LeafBlock, %LeafBlock914, %NodeBlock, %comb_entry.fragment, %old_entry, %bb_init
	switch i32 0, label %old_entry [
		 i32 2739, label %invcont28.fragment
		 i32 2688, label %meshBB28.fragment
		 i32 1318, label %meshBB32.fragment
		 i32 2964, label %unwind53.fragment.fragment
		 i32 824, label %unwind78.fragment.fragment
		 i32 1983, label %meshBB33.fragment
		 i32 2582, label %invcont30.fragment
		 i32 2235, label %meshBB36.fragment
		 i32 1275, label %meshBB343
		 i32 2719, label %invcont.fragment
		 i32 1500, label %entry1.fragment.fragment
		 i32 815, label %unwind11.fragment
		 i32 1051, label %entry
		 i32 2342, label %unwind
		 i32 1814, label %invcont
		 i32 315, label %invcont.unwind10_crit_edge
		 i32 2422, label %unwind10
		 i32 2663, label %unwind10.Unwind_crit_edge
		 i32 266, label %invcont14
		 i32 367, label %invcont14.unwind10_crit_edge
		 i32 2242, label %invcont15
		 i32 452, label %unwind20
		 i32 419, label %invcont.cond_next_crit_edge
		 i32 181, label %cond_true
		 i32 2089, label %unwind20.Unwind_crit_edge
		 i32 633, label %filter
		 i32 455, label %Unwind
		 i32 2016, label %entry1
		 i32 263, label %invcont33.unwind_crit_edge
		 i32 2498, label %invcont36
		 i32 2992, label %unwind37
		 i32 616, label %entry.unwind_crit_edge
		 i32 622, label %unwind11
		 i32 875, label %invcont16
		 i32 766, label %unwind53.nofilter_crit_edge
		 i32 668, label %filter62
		 i32 2138, label %unwind66
		 i32 713, label %unwind66.Unwind_crit_edge
		 i32 1422, label %invcont70
		 i32 1976, label %cond_true.unwind_crit_edge
		 i32 1263, label %invcont12
		 i32 2453, label %invcont12.unwind_crit_edge
		 i32 2876, label %invcont12.cond_next_crit_edge
		 i32 2271, label %cond_next
		 i32 2938, label %cond_next.unwind_crit_edge
		 i32 1082, label %invcont17
		 i32 531, label %invcont17.unwind_crit_edge
		 i32 111, label %invcont22
		 i32 1935, label %invcont22.unwind_crit_edge
		 i32 2004, label %invcont25
		 i32 1725, label %invcont25.unwind_crit_edge
		 i32 1701, label %invcont28
		 i32 957, label %invcont28.unwind_crit_edge
		 i32 165, label %invcont30
		 i32 899, label %invcont30.unwind_crit_edge
		 i32 1092, label %invcont33
		 i32 2869, label %unwind37.nofilter_crit_edge
		 i32 203, label %invcont41
		 i32 693, label %unwind43
		 i32 2895, label %unwind43.nofilter_crit_edge
		 i32 1174, label %invcont47
		 i32 1153, label %filter19
		 i32 2304, label %nofilter
		 i32 848, label %nofilter.Unwind_crit_edge
		 i32 1207, label %unwind53
		 i32 2848, label %filter75
		 i32 59, label %unwind78
		 i32 1213, label %unwind78.Unwind_crit_edge
		 i32 2199, label %filter87
		 i32 1268, label %Unwind20
		 i32 743, label %old_entry
		 i32 1276, label %meshBB319
		 i32 1619, label %meshBB320
		 i32 2047, label %meshBB331
		 i32 2828, label %meshBB23.fragment
		 i32 2530, label %meshBB332
		 i32 1389, label %meshBB318
		 i32 1450, label %meshBB317
		 i32 1416, label %meshBB31.fragment
		 i32 82, label %meshBB322
		 i32 853, label %unwind78.fragment316
		 i32 107, label %meshBB24.fragment
		 i32 1200, label %meshBB37.fragment
		 i32 605, label %unwind53.fragment314
		 i32 209, label %meshBB29.fragment
		 i32 1513, label %meshBB27.fragment
		 i32 1542, label %meshBB35.fragment
		 i32 1873, label %meshBB348
		 i32 472, label %meshBB325
		 i32 2615, label %meshBB22.fragment
		 i32 359, label %meshBB.fragment
		 i32 2467, label %Unwind20.fragment
		 i32 1671, label %unwind66.fragment.fragment
		 i32 1006, label %meshBB25.fragment
		 i32 1243, label %meshBB333
		 i32 2795, label %unwind43.fragment313
		 i32 1591, label %meshBB335
		 i32 773, label %meshBB341
		 i32 2440, label %cond_next.fragment
		 i32 487, label %meshBB326
		 i32 394, label %meshBB324
		 i32 14, label %invcont16.fragment
		 i32 574, label %entry1.fragment312
		 i32 1453, label %meshBB35
		 i32 345, label %entry1.fragment
		 i32 2951, label %unwind20.fragment
		 i32 1960, label %meshBB31
		 i32 2163, label %meshBB32
		 i32 1978, label %Unwind.fragment
		 i32 1559, label %unwind20.fragment.fragment
		 i32 950, label %unwind10.fragment
		 i32 1724, label %unwind53.fragment
		 i32 514, label %meshBB36
		 i32 1928, label %unwind10.fragment.fragment
		 i32 1266, label %meshBB26
		 i32 3148, label %unwind20.fragment311
		 i32 1581, label %unwind43.fragment
		 i32 1829, label %meshBB34
		 i32 1472, label %meshBB28
		 i32 2657, label %unwind66.fragment
		 i32 2169, label %meshBB22
		 i32 2619, label %meshBB
		 i32 1397, label %entry.fragment
		 i32 231, label %invcont41.fragment
		 i32 2557, label %meshBB338
		 i32 2387, label %meshBB30.fragment
		 i32 2927, label %meshBB340
		 i32 2331, label %meshBB321
		 i32 47, label %meshBB328
		 i32 1753, label %meshBB342
		 i32 2074, label %meshBB323
		 i32 2128, label %meshBB334
		 i32 2396, label %meshBB337
		 i32 1811, label %meshBB29
		 i32 1113, label %meshBB27
		 i32 2232, label %unwind10.fragment310
		 i32 804, label %meshBB24
		 i32 3099, label %meshBB30
		 i32 564, label %meshBB33
		 i32 1359, label %unwind.fragment
		 i32 1906, label %entry.fragment309
		 i32 2644, label %entry.fragment.fragment
		 i32 134, label %entry1.fragment.normaldest
		 i32 2767, label %comb_entry.fragment
		 i32 2577, label %meshBB25
		 i32 3128, label %meshBB37
		 i32 2360, label %meshBB23
		 i32 286, label %unwind78.fragment
		 i32 976, label %meshBB346
		 i32 2412, label %meshBB339
		 i32 876, label %meshBB345
		 i32 3078, label %meshBB329
		 i32 1297, label %meshBB347
		 i32 3051, label %meshBB336
		 i32 1342, label %meshBB344
		 i32 728, label %meshBB330
		 i32 1778, label %meshBB349
		 i32 2784, label %meshBB327
		 i32 1854, label %meshBB26.fragment
		 i32 1025, label %meshBB34.fragment
		 i32 2139, label %unwind43.fragment.fragment
		 i32 2217, label %nofilter.fragment
		 i32 665, label %invcont12.fragment
		 i32 316, label %invcont22.fragment
		 i32 1467, label %unwind66.fragment315
		 i32 3018, label %unwind37.fragment
		 i32 1123, label %invcont17.normaldest
		 i32 2104, label %NewDefault
		 i32 1639, label %LeafBlock
		 i32 925, label %LeafBlock914
		 i32 2880, label %NodeBlock
	]

old_entry:		; preds = %bb_main, %bb_main
	br label %bb_main

comb_entry.fragment:		; preds = %bb_main
	br label %bb_main

NodeBlock:		; preds = %bb_main
	br label %bb_main

LeafBlock914:		; preds = %bb_main
	br label %bb_main

LeafBlock:		; preds = %bb_main
	br label %bb_main

NewDefault:		; preds = %bb_main
	br label %bb_main

entry:		; preds = %bb_main
	br label %bb_main

entry.fragment309:		; preds = %bb_main
	br label %bb_main

entry.fragment:		; preds = %bb_main
	br label %bb_main

entry.fragment.fragment:		; preds = %bb_main
	invoke void @___ZNSt12_Vector_baseI6BigIntSaIS0_EE13_M_deallocateEPS0_j___ZNSt12_Vector_baseI6BigIntSaIS0_EED2Ev___ZNSt6vectorI6BigIntSaIS0_EEC1ERKS1_( %"struct.std::_Vector_base<BigInt,std::allocator<BigInt> >"* null, i32 28, %struct.BigInt* null, i32 0, %"struct.std::vector<BigInt,std::allocator<BigInt> >"* null, %struct.__false_type* null )
			to label %meshBB340 unwind label %meshBB325

unwind:		; preds = %bb_main
	br label %bb_main

unwind.fragment:		; preds = %bb_main
	br label %bb_main

invcont:		; preds = %bb_main
	br label %bb_main

invcont.fragment:		; preds = %bb_main
	invoke void @_ZN9Fibonacci10get_numberEj( %struct.BigInt* null sret , %struct.Fibonacci* %this_this, i32 %n_i_n_i )
			to label %invcont14 unwind label %meshBB37

invcont.unwind10_crit_edge:		; preds = %bb_main
	br label %bb_main

unwind10:		; preds = %bb_main
	br label %bb_main

unwind10.fragment310:		; preds = %bb_main
	br label %bb_main

unwind10.fragment:		; preds = %bb_main
	br label %bb_main

unwind10.fragment.fragment:		; preds = %bb_main
	invoke void @___ZNSt6vectorI6BigIntSaIS0_EED1Ev___ZN9FibonacciD1Ev___ZNSt6vectorImSaImEEC1ERKS0_( i32 57, %"struct.std::vector<BigInt,std::allocator<BigInt> >"* null, %struct.Fibonacci* null, %"struct.std::vector<ulong,std::allocator<ulong> >"* null, %struct.__false_type* null )
			to label %meshBB329 unwind label %meshBB24

unwind10.Unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont14:		; preds = %invcont.fragment, %bb_main
	br label %bb_main

invcont14.normaldest:		; No predecessors!
	invoke %"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"* @___ZN9__gnu_cxx17__normal_iteratorIP6BigIntSt6vectorIS1_SaIS1_EEEppEv___ZNSt6vectorImSaImEED1Ev___ZN6BigIntD1Ev___ZN9__gnu_cxx13new_allocatorI6BigIntE7destroyEPS1____ZSt8_DestroyIP6BigIntSaIS0_EEvT_S3_T0_( i32 14, %"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"* null, %"struct.std::vector<ulong,std::allocator<ulong> >"* null, %struct.BigInt* null, %struct.__false_type* null, %struct.BigInt* null, %struct.__false_type* null noalias  )
			to label %invcont15 unwind label %meshBB345		; <%"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"*>:0 [#uses=0]

invcont14.unwind10_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont15:		; preds = %invcont14.normaldest, %bb_main
	br label %bb_main

invcont15.normaldest:		; No predecessors!
	br label %UnifiedReturnBlock

unwind20:		; preds = %bb_main
	br label %bb_main

unwind20.fragment311:		; preds = %bb_main
	br label %bb_main

unwind20.fragment:		; preds = %bb_main
	br label %bb_main

unwind20.fragment.fragment:		; preds = %bb_main
	br label %bb_main

unwind20.Unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

filter:		; preds = %bb_main
	br label %UnifiedUnreachableBlock

Unwind:		; preds = %bb_main
	br label %bb_main

Unwind.fragment:		; preds = %bb_main
	br label %UnifiedUnreachableBlock

entry1:		; preds = %bb_main
	br label %bb_main

entry1.fragment312:		; preds = %bb_main
	br label %bb_main

entry1.fragment:		; preds = %bb_main
	br label %bb_main

entry1.fragment.fragment:		; preds = %bb_main
	%tmp52 = invoke i32 @___ZN9__gnu_cxxmiIPK6BigIntS3_St6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T1_E15difference_typeERKSA_RKNS7_IT0_S9_EE___ZNKSt6vectorI6BigIntSaIS0_EE4sizeEv___ZNK9Fibonacci16show_all_numbersEv___ZNKSt6vectorI6BigIntSaIS0_EE8capacityEv( %"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"* null, %"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"* null, %"struct.std::vector<BigInt,std::allocator<BigInt> >"* null, i32 16, %struct.Fibonacci* null )
			to label %entry1.fragment.normaldest unwind label %meshBB320		; <i32> [#uses=0]

entry.unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

unwind11:		; preds = %bb_main
	br label %bb_main

unwind11.fragment:		; preds = %bb_main
	br label %bb_main

invcont16:		; preds = %bb_main
	br label %bb_main

invcont16.fragment:		; preds = %bb_main
	br label %bb_main

invcont.cond_next_crit_edge:		; preds = %bb_main
	br label %bb_main

cond_true:		; preds = %bb_main
	invoke void @_ZN9Fibonacci10get_numberEj( %struct.BigInt* null sret , %struct.Fibonacci* %this_this, i32 %n_i_n_i )
			to label %meshBB323 unwind label %cond_true.unwind_crit_edge

cond_true.unwind_crit_edge:		; preds = %cond_true, %bb_main
	br label %bb_main

cond_true.unwind_crit_edge.unwinddest:		; No predecessors!
	br label %bb_main

invcont12:		; preds = %bb_main
	br label %bb_main

invcont12.fragment:		; preds = %bb_main
	invoke %"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"* @___ZN9__gnu_cxx17__normal_iteratorIP6BigIntSt6vectorIS1_SaIS1_EEEppEv___ZNSt6vectorImSaImEED1Ev___ZN6BigIntD1Ev___ZN9__gnu_cxx13new_allocatorI6BigIntE7destroyEPS1____ZSt8_DestroyIP6BigIntSaIS0_EEvT_S3_T0_( i32 14, %"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"* null, %"struct.std::vector<ulong,std::allocator<ulong> >"* null, %struct.BigInt* null, %struct.__false_type* null, %struct.BigInt* null, %struct.__false_type* null noalias  )
			to label %meshBB30 unwind label %meshBB337		; <%"struct.__gnu_cxx::__normal_iterator<BigInt*,std::vector<BigInt, std::allocator<BigInt> > >"*>:1 [#uses=0]

invcont12.unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont12.cond_next_crit_edge:		; preds = %bb_main
	br label %bb_main

cond_next:		; preds = %bb_main
	br label %bb_main

cond_next.fragment:		; preds = %bb_main
	%tmp183 = invoke %struct.BigInt* @___ZNSt6vectorI6BigIntSaIS0_EEixEj___ZNSt6vectorI6BigIntSaIS0_EE3endEv( %"struct.std::vector<BigInt,std::allocator<BigInt> >"* null, i32 %n_i_n_i, i32 29 )
			to label %invcont17 unwind label %meshBB336		; <%struct.BigInt*> [#uses=0]

cond_next.unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont17:		; preds = %cond_next.fragment, %bb_main
	br label %bb_main

invcont17.normaldest917:		; No predecessors!
	%tmp23 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc( %"struct.std::basic_ostream<char,std::char_traits<char> >"* null, i8* getelementptr ([6 x i8]* @.str13, i32 0, i32 0) )
			to label %invcont17.normaldest unwind label %meshBB318		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=1]

invcont17.unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont22:		; preds = %bb_main
	br label %bb_main

invcont22.fragment:		; preds = %bb_main
	%tmp26 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZNSolsEm( %"struct.std::basic_ostream<char,std::char_traits<char> >"* undef, i32 %n_i_n_i )
			to label %invcont25 unwind label %meshBB319		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=1]

invcont22.unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont25:		; preds = %invcont22.fragment, %bb_main
	br label %bb_main

invcont25.normaldest:		; No predecessors!
	%tmp2918 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc( %"struct.std::basic_ostream<char,std::char_traits<char> >"* %tmp26, i8* getelementptr ([5 x i8]* @.str14, i32 0, i32 0) )
			to label %invcont28 unwind label %invcont25.unwind_crit_edge		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=0]

invcont25.unwind_crit_edge:		; preds = %invcont25.normaldest, %bb_main
	br label %bb_main

invcont25.unwind_crit_edge.unwinddest:		; No predecessors!
	br label %bb_main

invcont28:		; preds = %invcont25.normaldest, %bb_main
	br label %bb_main

invcont28.normaldest:		; No predecessors!
	br label %bb_main

invcont28.fragment:		; preds = %bb_main
	%tmp311 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @___ZlsRSoRK6BigInt___ZN9__gnu_cxx13new_allocatorI6BigIntE10deallocateEPS1_j( i32 32, %"struct.std::basic_ostream<char,std::char_traits<char> >"* undef, %struct.BigInt* undef, %struct.__false_type* null, i32 0 )
			to label %invcont30 unwind label %meshBB322		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=0]

invcont28.unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont30:		; preds = %invcont28.fragment, %bb_main
	br label %bb_main

invcont30.normaldest:		; No predecessors!
	br label %bb_main

invcont30.fragment:		; preds = %bb_main
	%tmp34 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc( %"struct.std::basic_ostream<char,std::char_traits<char> >"* undef, i8* getelementptr ([2 x i8]* @.str15, i32 0, i32 0) )
			to label %meshBB26 unwind label %invcont30.unwind_crit_edge		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=0]

invcont30.unwind_crit_edge:		; preds = %invcont30.fragment, %bb_main
	br label %bb_main

invcont30.unwind_crit_edge.unwinddest:		; No predecessors!
	br label %bb_main

invcont33:		; preds = %bb_main
	invoke void @_ZNKSt19basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv( %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* null sret , %"struct.std::ostringstream"* null )
			to label %invcont36 unwind label %invcont33.unwind_crit_edge

invcont33.unwind_crit_edge:		; preds = %invcont33, %bb_main
	br label %bb_main

invcont33.unwind_crit_edge.unwinddest:		; No predecessors!
	br label %bb_main

invcont36:		; preds = %invcont33, %bb_main
	br label %bb_main

invcont36.normaldest:		; No predecessors!
	%tmp42 = invoke %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E( %"struct.std::basic_ostream<char,std::char_traits<char> >"* @_ZSt4cout, %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* null )
			to label %invcont41 unwind label %meshBB338		; <%"struct.std::basic_ostream<char,std::char_traits<char> >"*> [#uses=0]

unwind37:		; preds = %bb_main
	br label %bb_main

unwind37.fragment:		; preds = %bb_main
	invoke void @_ZNSsD1Ev( %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* null )
			to label %meshBB330 unwind label %meshBB22

unwind37.nofilter_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont41:		; preds = %invcont36.normaldest, %bb_main
	br label %bb_main

invcont41.normaldest:		; No predecessors!
	br label %bb_main

invcont41.fragment:		; preds = %bb_main
	invoke void @_ZNSsD1Ev( %"struct.std::basic_string<char,std::char_traits<char>,std::allocator<char> >"* null )
			to label %meshBB23 unwind label %meshBB29

unwind43:		; preds = %bb_main
	br label %bb_main

unwind43.fragment313:		; preds = %bb_main
	br label %bb_main

unwind43.fragment:		; preds = %bb_main
	br label %bb_main

unwind43.fragment.fragment:		; preds = %bb_main
	br label %bb_main

unwind43.nofilter_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont47:		; preds = %bb_main
	invoke void @_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev( %"struct.std::ostringstream"* null )
			to label %invcont70 unwind label %meshBB28

filter19:		; preds = %bb_main
	br label %UnifiedUnreachableBlock

nofilter:		; preds = %bb_main
	br label %bb_main

nofilter.fragment:		; preds = %bb_main
	invoke void @_ZNSt19basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev( %"struct.std::ostringstream"* null )
			to label %nofilter.Unwind_crit_edge unwind label %meshBB

nofilter.Unwind_crit_edge:		; preds = %nofilter.fragment, %bb_main
	br label %bb_main

nofilter.Unwind_crit_edge.normaldest:		; No predecessors!
	br label %bb_main

unwind53:		; preds = %bb_main
	br label %bb_main

unwind53.fragment314:		; preds = %bb_main
	br label %bb_main

unwind53.fragment:		; preds = %bb_main
	br label %bb_main

unwind53.fragment.fragment:		; preds = %bb_main
	br label %bb_main

unwind53.nofilter_crit_edge:		; preds = %bb_main
	br label %bb_main

filter62:		; preds = %bb_main
	br label %UnifiedUnreachableBlock

unwind66:		; preds = %bb_main
	br label %bb_main

unwind66.fragment315:		; preds = %bb_main
	br label %bb_main

unwind66.fragment:		; preds = %bb_main
	br label %bb_main

unwind66.fragment.fragment:		; preds = %bb_main
	br label %bb_main

unwind66.Unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

invcont70:		; preds = %invcont47, %bb_main
	br label %bb_main

invcont70.normaldest:		; No predecessors!
	br label %UnifiedReturnBlock

filter75:		; preds = %bb_main
	br label %UnifiedUnreachableBlock

unwind78:		; preds = %bb_main
	br label %bb_main

unwind78.fragment316:		; preds = %bb_main
	br label %bb_main

unwind78.fragment:		; preds = %bb_main
	br label %bb_main

unwind78.fragment.fragment:		; preds = %bb_main
	br label %bb_main

unwind78.Unwind_crit_edge:		; preds = %bb_main
	br label %bb_main

filter87:		; preds = %bb_main
	br label %UnifiedUnreachableBlock

Unwind20:		; preds = %bb_main
	br label %bb_main

Unwind20.fragment:		; preds = %bb_main
	br label %UnifiedUnreachableBlock

meshBB:		; preds = %nofilter.fragment, %bb_main
	br label %bb_main

meshBB.unwinddest:		; No predecessors!
	br label %bb_main

meshBB.fragment:		; preds = %bb_main
	br label %bb_main

meshBB22:		; preds = %unwind37.fragment, %bb_main
	br label %bb_main

meshBB22.unwinddest:		; No predecessors!
	br label %bb_main

meshBB22.fragment:		; preds = %bb_main
	br label %bb_main

entry1.fragment.normaldest:		; preds = %entry1.fragment.fragment, %bb_main
	br label %bb_main

entry1.fragment.normaldest.normaldest:		; No predecessors!
	br label %bb_main

meshBB23:		; preds = %invcont41.fragment, %bb_main
	br label %bb_main

meshBB23.normaldest:		; No predecessors!
	br label %bb_main

meshBB23.fragment:		; preds = %bb_main
	br label %bb_main

meshBB24:		; preds = %unwind10.fragment.fragment, %bb_main
	br label %bb_main

meshBB24.unwinddest:		; No predecessors!
	br label %bb_main

meshBB24.fragment:		; preds = %bb_main
	br label %bb_main

meshBB25:		; preds = %bb_main
	br label %bb_main

meshBB25.fragment:		; preds = %bb_main
	br label %bb_main

meshBB26:		; preds = %invcont30.fragment, %bb_main
	br label %bb_main

meshBB26.normaldest:		; No predecessors!
	br label %bb_main

meshBB26.fragment:		; preds = %bb_main
	br label %bb_main

meshBB27:		; preds = %bb_main
	br label %bb_main

meshBB27.fragment:		; preds = %bb_main
	br label %bb_main

meshBB28:		; preds = %invcont47, %bb_main
	br label %bb_main

meshBB28.unwinddest:		; No predecessors!
	br label %bb_main

meshBB28.fragment:		; preds = %bb_main
	br label %bb_main

meshBB29:		; preds = %invcont41.fragment, %bb_main
	br label %bb_main

meshBB29.unwinddest:		; No predecessors!
	br label %bb_main

meshBB29.fragment:		; preds = %bb_main
	br label %bb_main

meshBB30:		; preds = %invcont12.fragment, %bb_main
	br label %bb_main

meshBB30.normaldest:		; No predecessors!
	br label %bb_main

meshBB30.fragment:		; preds = %bb_main
	br label %bb_main

meshBB31:		; preds = %bb_main
	br label %bb_main

meshBB31.fragment:		; preds = %bb_main
	br label %bb_main

meshBB32:		; preds = %bb_main
	br label %bb_main

meshBB32.fragment:		; preds = %bb_main
	br label %bb_main

meshBB33:		; preds = %bb_main
	br label %bb_main

meshBB33.fragment:		; preds = %bb_main
	br label %bb_main

meshBB34:		; preds = %bb_main
	br label %bb_main

meshBB34.fragment:		; preds = %bb_main
	br label %bb_main

meshBB35:		; preds = %bb_main
	br label %bb_main

meshBB35.fragment:		; preds = %bb_main
	br label %bb_main

meshBB36:		; preds = %bb_main
	br label %bb_main

meshBB36.fragment:		; preds = %bb_main
	br label %bb_main

meshBB37:		; preds = %invcont.fragment, %bb_main
	br label %bb_main

meshBB37.unwinddest:		; No predecessors!
	br label %bb_main

meshBB37.fragment:		; preds = %bb_main
	br label %bb_main

meshBB317:		; preds = %bb_main
	br label %bb_main

meshBB318:		; preds = %invcont17.normaldest917, %bb_main
	br label %bb_main

meshBB318.unwinddest:		; No predecessors!
	br label %bb_main

meshBB319:		; preds = %invcont22.fragment, %bb_main
	br label %bb_main

meshBB319.unwinddest:		; No predecessors!
	br label %bb_main

meshBB320:		; preds = %entry1.fragment.fragment, %bb_main
	br label %bb_main

meshBB320.unwinddest:		; No predecessors!
	br label %bb_main

meshBB321:		; preds = %bb_main
	br label %bb_main

meshBB322:		; preds = %invcont28.fragment, %bb_main
	br label %bb_main

meshBB322.unwinddest:		; No predecessors!
	br label %bb_main

meshBB323:		; preds = %cond_true, %bb_main
	br label %bb_main

meshBB323.normaldest:		; No predecessors!
	br label %bb_main

meshBB324:		; preds = %bb_main
	br label %bb_main

meshBB325:		; preds = %entry.fragment.fragment, %bb_main
	br label %bb_main

meshBB325.unwinddest:		; No predecessors!
	br label %bb_main

meshBB326:		; preds = %bb_main
	br label %bb_main

meshBB327:		; preds = %bb_main
	br label %bb_main

meshBB328:		; preds = %bb_main
	br label %bb_main

meshBB329:		; preds = %unwind10.fragment.fragment, %bb_main
	br label %bb_main

meshBB329.normaldest:		; No predecessors!
	br label %bb_main

meshBB330:		; preds = %unwind37.fragment, %bb_main
	br label %bb_main

meshBB330.normaldest:		; No predecessors!
	br label %bb_main

meshBB331:		; preds = %bb_main
	br label %bb_main

meshBB332:		; preds = %bb_main
	br label %bb_main

meshBB333:		; preds = %bb_main
	br label %bb_main

meshBB334:		; preds = %bb_main
	br label %bb_main

meshBB335:		; preds = %bb_main
	br label %bb_main

meshBB336:		; preds = %cond_next.fragment, %bb_main
	br label %bb_main

meshBB336.unwinddest:		; No predecessors!
	br label %bb_main

meshBB337:		; preds = %invcont12.fragment, %bb_main
	br label %bb_main

meshBB337.unwinddest:		; No predecessors!
	br label %bb_main

meshBB338:		; preds = %invcont36.normaldest, %bb_main
	br label %bb_main

meshBB338.unwinddest:		; No predecessors!
	br label %bb_main

invcont17.normaldest:		; preds = %invcont17.normaldest917, %bb_main
	br label %bb_main

invcont17.normaldest.normaldest:		; No predecessors!
	store %"struct.std::basic_ostream<char,std::char_traits<char> >"* %tmp23, %"struct.std::basic_ostream<char,std::char_traits<char> >"** undef
	br label %bb_main

meshBB339:		; preds = %bb_main
	br label %bb_main

meshBB340:		; preds = %entry.fragment.fragment, %bb_main
	br label %bb_main

meshBB340.normaldest:		; No predecessors!
	br label %bb_main

meshBB341:		; preds = %bb_main
	br label %bb_main

meshBB342:		; preds = %bb_main
	br label %bb_main

meshBB343:		; preds = %bb_main
	br label %bb_main

meshBB344:		; preds = %bb_main
	br label %bb_main

meshBB345:		; preds = %invcont14.normaldest, %bb_main
	br label %bb_main

meshBB345.unwinddest:		; No predecessors!
	br label %bb_main

meshBB346:		; preds = %bb_main
	br label %bb_main

meshBB347:		; preds = %bb_main
	br label %bb_main

meshBB348:		; preds = %bb_main
	br label %bb_main

meshBB349:		; preds = %bb_main
	br label %bb_main

UnifiedUnreachableBlock:		; preds = %Unwind20.fragment, %filter87, %filter75, %filter62, %filter19, %Unwind.fragment, %filter
	unreachable

UnifiedReturnBlock:		; preds = %invcont70.normaldest, %invcont15.normaldest
	ret void
}
