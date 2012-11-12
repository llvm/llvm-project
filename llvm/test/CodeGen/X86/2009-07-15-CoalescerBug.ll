; RUN: llc < %s -mtriple=x86_64-apple-darwin10

	%struct.ANY = type { i8* }
	%struct.AV = type { %struct.XPVAV*, i32, i32 }
	%struct.CLONE_PARAMS = type { %struct.AV*, i64, %struct.PerlInterpreter* }
	%struct.CV = type { %struct.XPVCV*, i32, i32 }
	%struct.DIR = type { i32, i64, i64, i8*, i32, i64, i64, i32, %struct.__darwin_pthread_mutex_t, %struct._telldir* }
	%struct.GP = type { %struct.SV*, i32, %struct.io*, %struct.CV*, %struct.AV*, %struct.HV*, %struct.GV*, %struct.CV*, i32, i32, i32, i8* }
	%struct.GV = type { %struct.XPVGV*, i32, i32 }
	%struct.HE = type { %struct.HE*, %struct.HEK*, %struct.SV* }
	%struct.HEK = type { i32, i32, [1 x i8] }
	%struct.HV = type { %struct.XPVHV*, i32, i32 }
	%struct.MAGIC = type { %struct.MAGIC*, %struct.MGVTBL*, i16, i8, i8, %struct.SV*, i8*, i32 }
	%struct.MGVTBL = type { i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*)*, i32 (%struct.SV*, %struct.MAGIC*, %struct.SV*, i8*, i32)*, i32 (%struct.MAGIC*, %struct.CLONE_PARAMS*)* }
	%struct.OP = type { %struct.OP*, %struct.OP*, %struct.OP* ()*, i64, i16, i16, i8, i8 }
	%struct.PMOP = type { %struct.OP*, %struct.OP*, %struct.OP* ()*, i64, i16, i16, i8, i8, %struct.OP*, %struct.OP*, %struct.OP*, %struct.OP*, %struct.PMOP*, %struct.REGEXP*, i32, i32, i8, %struct.HV* }
	%struct.PerlIO_funcs = type { i64, i8*, i64, i32, i64 (%struct.PerlIOl**, i8*, %struct.SV*, %struct.PerlIO_funcs*)*, i64 (%struct.PerlIOl**)*, %struct.PerlIOl** (%struct.PerlIO_funcs*, %struct.PerlIO_list_t*, i64, i8*, i32, i32, i32, %struct.PerlIOl**, i32, %struct.SV**)*, i64 (%struct.PerlIOl**)*, %struct.SV* (%struct.PerlIOl**, %struct.CLONE_PARAMS*, i32)*, i64 (%struct.PerlIOl**)*, %struct.PerlIOl** (%struct.PerlIOl**, %struct.PerlIOl**, %struct.CLONE_PARAMS*, i32)*, i64 (%struct.PerlIOl**, i8*, i64)*, i64 (%struct.PerlIOl**, i8*, i64)*, i64 (%struct.PerlIOl**, i8*, i64)*, i64 (%struct.PerlIOl**, i64, i32)*, i64 (%struct.PerlIOl**)*, i64 (%struct.PerlIOl**)*, i64 (%struct.PerlIOl**)*, i64 (%struct.PerlIOl**)*, i64 (%struct.PerlIOl**)*, i64 (%struct.PerlIOl**)*, void (%struct.PerlIOl**)*, void (%struct.PerlIOl**)*, i8* (%struct.PerlIOl**)*, i64 (%struct.PerlIOl**)*, i8* (%struct.PerlIOl**)*, i64 (%struct.PerlIOl**)*, void (%struct.PerlIOl**, i8*, i64)* }
	%struct.PerlIO_list_t = type { i64, i64, i64, %struct.PerlIO_pair_t* }
	%struct.PerlIO_pair_t = type { %struct.PerlIO_funcs*, %struct.SV* }
	%struct.PerlIOl = type { %struct.PerlIOl*, %struct.PerlIO_funcs*, i32 }
	%struct.PerlInterpreter = type { i8 }
	%struct.REGEXP = type { i32*, i32*, %struct.regnode*, %struct.reg_substr_data*, i8*, %struct.reg_data*, i8*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, [1 x %struct.regnode] }
	%struct.SV = type { i8*, i32, i32 }
	%struct.XPVAV = type { i8*, i64, i64, i64, double, %struct.MAGIC*, %struct.HV*, %struct.SV**, %struct.SV*, i8 }
	%struct.XPVCV = type { i8*, i64, i64, i64, double, %struct.MAGIC*, %struct.HV*, %struct.HV*, %struct.OP*, %struct.OP*, void (%struct.CV*)*, %struct.ANY, %struct.GV*, i8*, i64, %struct.AV*, %struct.CV*, i16, i32 }
	%struct.XPVGV = type { i8*, i64, i64, i64, double, %struct.MAGIC*, %struct.HV*, %struct.GP*, i8*, i64, %struct.HV*, i8 }
	%struct.XPVHV = type { i8*, i64, i64, i64, double, %struct.MAGIC*, %struct.HV*, i32, %struct.HE*, %struct.PMOP*, i8* }
	%struct.XPVIO = type { i8*, i64, i64, i64, double, %struct.MAGIC*, %struct.HV*, %struct.PerlIOl**, %struct.PerlIOl**, %struct.anon, i64, i64, i64, i64, i8*, %struct.GV*, i8*, %struct.GV*, i8*, %struct.GV*, i16, i8, i8 }
	%struct.__darwin_pthread_mutex_t = type { i64, [56 x i8] }
	%struct._telldir = type opaque
	%struct.anon = type { %struct.DIR* }
	%struct.io = type { %struct.XPVIO*, i32, i32 }
	%struct.reg_data = type { i32, i8*, [1 x i8*] }
	%struct.reg_substr_data = type { [3 x %struct.reg_substr_datum] }
	%struct.reg_substr_datum = type { i32, i32, %struct.SV*, %struct.SV* }
	%struct.regnode = type { i8, i8, i16 }

define i32 @Perl_yylex() nounwind ssp {
entry:
	br i1 undef, label %bb21, label %bb

bb:		; preds = %entry
	unreachable

bb21:		; preds = %entry
	switch i32 undef, label %bb103 [
		i32 1, label %bb101
		i32 4, label %bb75
		i32 6, label %bb68
		i32 7, label %bb67
		i32 8, label %bb25
	]

bb25:		; preds = %bb21
	ret i32 41

bb67:		; preds = %bb21
	ret i32 40

bb68:		; preds = %bb21
	br i1 undef, label %bb69, label %bb70

bb69:		; preds = %bb68
	ret i32 undef

bb70:		; preds = %bb68
	unreachable

bb75:		; preds = %bb21
	unreachable

bb101:		; preds = %bb21
	unreachable

bb103:		; preds = %bb21
	switch i32 undef, label %bb104 [
		i32 0, label %bb126
		i32 4, label %fake_eof
		i32 26, label %fake_eof
		i32 34, label %bb1423
		i32 36, label %bb1050
		i32 37, label %bb534
		i32 39, label %bb1412
		i32 41, label %bb643
		i32 44, label %bb544
		i32 48, label %bb1406
		i32 49, label %bb1406
		i32 50, label %bb1406
		i32 51, label %bb1406
		i32 52, label %bb1406
		i32 53, label %bb1406
		i32 54, label %bb1406
		i32 55, label %bb1406
		i32 56, label %bb1406
		i32 57, label %bb1406
		i32 59, label %bb639
		i32 65, label %keylookup
		i32 66, label %keylookup
		i32 67, label %keylookup
		i32 68, label %keylookup
		i32 69, label %keylookup
		i32 70, label %keylookup
		i32 71, label %keylookup
		i32 72, label %keylookup
		i32 73, label %keylookup
		i32 74, label %keylookup
		i32 75, label %keylookup
		i32 76, label %keylookup
		i32 77, label %keylookup
		i32 78, label %keylookup
		i32 79, label %keylookup
		i32 80, label %keylookup
		i32 81, label %keylookup
		i32 82, label %keylookup
		i32 83, label %keylookup
		i32 84, label %keylookup
		i32 85, label %keylookup
		i32 86, label %keylookup
		i32 87, label %keylookup
		i32 88, label %keylookup
		i32 89, label %keylookup
		i32 90, label %keylookup
		i32 92, label %bb1455
		i32 95, label %keylookup
		i32 96, label %bb1447
		i32 97, label %keylookup
		i32 98, label %keylookup
		i32 99, label %keylookup
		i32 100, label %keylookup
		i32 101, label %keylookup
		i32 102, label %keylookup
		i32 103, label %keylookup
		i32 104, label %keylookup
		i32 105, label %keylookup
		i32 106, label %keylookup
		i32 107, label %keylookup
		i32 108, label %keylookup
		i32 109, label %keylookup
		i32 110, label %keylookup
		i32 111, label %keylookup
		i32 112, label %keylookup
		i32 113, label %keylookup
		i32 114, label %keylookup
		i32 115, label %keylookup
		i32 116, label %keylookup
		i32 117, label %keylookup
		i32 118, label %keylookup
		i32 119, label %keylookup
		i32 120, label %keylookup
		i32 121, label %keylookup
		i32 122, label %keylookup
		i32 126, label %bb544
	]

bb104:		; preds = %bb103
	unreachable

bb126:		; preds = %bb103
	ret i32 0

fake_eof:		; preds = %bb1841, %bb103, %bb103
	unreachable

bb534:		; preds = %bb103
	unreachable

bb544:		; preds = %bb103, %bb103
	ret i32 undef

bb639:		; preds = %bb103
	unreachable

bb643:		; preds = %bb103
	unreachable

bb1050:		; preds = %bb103
	unreachable

bb1406:		; preds = %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103
	unreachable

bb1412:		; preds = %bb103
	unreachable

bb1423:		; preds = %bb103
	unreachable

bb1447:		; preds = %bb103
	unreachable

bb1455:		; preds = %bb103
	unreachable

keylookup:		; preds = %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103, %bb103
	br i1 undef, label %bb1498, label %bb1496

bb1496:		; preds = %keylookup
	br i1 undef, label %bb1498, label %bb1510.preheader

bb1498:		; preds = %bb1496, %keylookup
	unreachable

bb1510.preheader:		; preds = %bb1496
	br i1 undef, label %bb1511, label %bb1518

bb1511:		; preds = %bb1510.preheader
	br label %bb1518

bb1518:		; preds = %bb1511, %bb1510.preheader
	switch i32 undef, label %bb741.i4285 [
		i32 95, label %bb744.i4287
		i32 115, label %bb852.i4394
	]

bb741.i4285:		; preds = %bb1518
	br label %Perl_keyword.exit4735

bb744.i4287:		; preds = %bb1518
	br label %Perl_keyword.exit4735

bb852.i4394:		; preds = %bb1518
	br i1 undef, label %bb861.i4404, label %bb856.i4399

bb856.i4399:		; preds = %bb852.i4394
	br label %Perl_keyword.exit4735

bb861.i4404:		; preds = %bb852.i4394
	br label %Perl_keyword.exit4735

Perl_keyword.exit4735:		; preds = %bb861.i4404, %bb856.i4399, %bb744.i4287, %bb741.i4285
	br i1 undef, label %bb1544, label %reserved_word

bb1544:		; preds = %Perl_keyword.exit4735
	br i1 undef, label %bb1565, label %bb1545

bb1545:		; preds = %bb1544
	br i1 undef, label %bb1563, label %bb1558

bb1558:		; preds = %bb1545
	%0 = load %struct.SV** undef		; <%struct.SV*> [#uses=1]
	%1 = bitcast %struct.SV* %0 to %struct.GV*		; <%struct.GV*> [#uses=5]
	br i1 undef, label %bb1563, label %bb1559

bb1559:		; preds = %bb1558
	br i1 undef, label %bb1560, label %bb1563

bb1560:		; preds = %bb1559
	br i1 undef, label %bb1563, label %bb1561

bb1561:		; preds = %bb1560
	br i1 undef, label %bb1562, label %bb1563

bb1562:		; preds = %bb1561
	br label %bb1563

bb1563:		; preds = %bb1562, %bb1561, %bb1560, %bb1559, %bb1558, %bb1545
	%gv19.3 = phi %struct.GV* [ %1, %bb1562 ], [ undef, %bb1545 ], [ %1, %bb1558 ], [ %1, %bb1559 ], [ %1, %bb1560 ], [ %1, %bb1561 ]		; <%struct.GV*> [#uses=0]
	br i1 undef, label %bb1565, label %reserved_word

bb1565:		; preds = %bb1563, %bb1544
	br i1 undef, label %bb1573, label %bb1580

bb1573:		; preds = %bb1565
	br label %bb1580

bb1580:		; preds = %bb1573, %bb1565
	br i1 undef, label %bb1595, label %reserved_word

bb1595:		; preds = %bb1580
	br i1 undef, label %reserved_word, label %bb1597

bb1597:		; preds = %bb1595
	br i1 undef, label %reserved_word, label %bb1602

bb1602:		; preds = %bb1597
	br label %reserved_word

reserved_word:		; preds = %bb1602, %bb1597, %bb1595, %bb1580, %bb1563, %Perl_keyword.exit4735
	switch i32 undef, label %bb2012 [
		i32 1, label %bb1819
		i32 2, label %bb1830
		i32 4, label %bb1841
		i32 5, label %bb1841
		i32 8, label %bb1880
		i32 14, label %bb1894
		i32 16, label %bb1895
		i32 17, label %bb1896
		i32 18, label %bb1897
		i32 19, label %bb1898
		i32 20, label %bb1899
		i32 22, label %bb1906
		i32 23, label %bb1928
		i32 24, label %bb2555
		i32 26, label %bb1929
		i32 31, label %bb1921
		i32 32, label %bb1930
		i32 33, label %bb1905
		i32 34, label %bb1936
		i32 35, label %bb1927
		i32 37, label %bb1962
		i32 40, label %bb1951
		i32 41, label %bb1946
		i32 42, label %bb1968
		i32 44, label %bb1969
		i32 45, label %bb1970
		i32 46, label %bb2011
		i32 47, label %bb2006
		i32 48, label %bb2007
		i32 49, label %bb2009
		i32 50, label %bb2010
		i32 51, label %bb2008
		i32 53, label %bb1971
		i32 54, label %bb1982
		i32 55, label %bb2005
		i32 59, label %bb2081
		i32 61, label %bb2087
		i32 64, label %bb2080
		i32 65, label %really_sub
		i32 66, label %bb2079
		i32 67, label %bb2089
		i32 69, label %bb2155
		i32 72, label %bb2137
		i32 74, label %bb2138
		i32 75, label %bb2166
		i32 76, label %bb2144
		i32 78, label %bb2145
		i32 81, label %bb2102
		i32 82, label %bb2108
		i32 84, label %bb2114
		i32 85, label %bb2115
		i32 86, label %bb2116
		i32 89, label %bb2146
		i32 90, label %bb2147
		i32 91, label %bb2148
		i32 93, label %bb2154
		i32 94, label %bb2167
		i32 96, label %bb2091
		i32 97, label %bb2090
		i32 98, label %bb2088
		i32 100, label %bb2173
		i32 101, label %bb2174
		i32 102, label %bb2175
		i32 103, label %bb2180
		i32 104, label %bb2181
		i32 106, label %bb2187
		i32 107, label %bb2188
		i32 110, label %bb2206
		i32 112, label %bb2217
		i32 113, label %bb2218
		i32 114, label %bb2199
		i32 119, label %bb2205
		i32 120, label %bb2229
		i32 121, label %bb2233
		i32 122, label %bb2234
		i32 123, label %bb2235
		i32 124, label %bb2236
		i32 125, label %bb2237
		i32 126, label %bb2238
		i32 127, label %bb2239
		i32 128, label %bb2268
		i32 129, label %bb2267
		i32 133, label %bb2276
		i32 134, label %bb2348
		i32 135, label %bb2337
		i32 137, label %bb2239
		i32 138, label %bb2367
		i32 139, label %bb2368
		i32 140, label %bb2369
		i32 141, label %bb2357
		i32 143, label %bb2349
		i32 144, label %bb2350
		i32 146, label %bb2356
		i32 147, label %bb2370
		i32 148, label %bb2445
		i32 149, label %bb2453
		i32 151, label %bb2381
		i32 152, label %bb2457
		i32 154, label %bb2516
		i32 156, label %bb2522
		i32 158, label %bb2527
		i32 159, label %bb2537
		i32 160, label %bb2503
		i32 162, label %bb2504
		i32 163, label %bb2464
		i32 165, label %bb2463
		i32 166, label %bb2538
		i32 168, label %bb2515
		i32 170, label %bb2549
		i32 172, label %bb2566
		i32 173, label %bb2595
		i32 174, label %bb2565
		i32 175, label %bb2567
		i32 176, label %bb2568
		i32 177, label %bb2569
		i32 178, label %bb2570
		i32 179, label %bb2594
		i32 182, label %bb2571
		i32 183, label %bb2572
		i32 185, label %bb2593
		i32 186, label %bb2583
		i32 187, label %bb2596
		i32 189, label %bb2602
		i32 190, label %bb2603
		i32 191, label %bb2604
		i32 192, label %bb2605
		i32 193, label %bb2606
		i32 196, label %bb2617
		i32 197, label %bb2618
		i32 198, label %bb2619
		i32 199, label %bb2627
		i32 200, label %bb2625
		i32 201, label %bb2626
		i32 206, label %really_sub
		i32 207, label %bb2648
		i32 208, label %bb2738
		i32 209, label %bb2739
		i32 210, label %bb2740
		i32 211, label %bb2742
		i32 212, label %bb2741
		i32 213, label %bb2737
		i32 214, label %bb2743
		i32 217, label %bb2758
		i32 219, label %bb2764
		i32 220, label %bb2765
		i32 221, label %bb2744
		i32 222, label %bb2766
		i32 226, label %bb2785
		i32 227, label %bb2783
		i32 228, label %bb2784
		i32 229, label %bb2790
		i32 230, label %bb2797
		i32 232, label %bb2782
		i32 234, label %bb2791
		i32 236, label %bb2815
		i32 237, label %bb2818
		i32 238, label %bb2819
		i32 239, label %bb2820
		i32 240, label %bb2817
		i32 241, label %bb2816
		i32 242, label %bb2821
		i32 243, label %bb2826
		i32 244, label %bb2829
		i32 245, label %bb2830
	]

bb1819:		; preds = %reserved_word
	unreachable

bb1830:		; preds = %reserved_word
	unreachable

bb1841:		; preds = %reserved_word, %reserved_word
	br i1 undef, label %fake_eof, label %bb1842

bb1842:		; preds = %bb1841
	unreachable

bb1880:		; preds = %reserved_word
	unreachable

bb1894:		; preds = %reserved_word
	ret i32 undef

bb1895:		; preds = %reserved_word
	ret i32 301

bb1896:		; preds = %reserved_word
	ret i32 undef

bb1897:		; preds = %reserved_word
	ret i32 undef

bb1898:		; preds = %reserved_word
	ret i32 undef

bb1899:		; preds = %reserved_word
	ret i32 undef

bb1905:		; preds = %reserved_word
	ret i32 278

bb1906:		; preds = %reserved_word
	unreachable

bb1921:		; preds = %reserved_word
	ret i32 288

bb1927:		; preds = %reserved_word
	ret i32 undef

bb1928:		; preds = %reserved_word
	ret i32 undef

bb1929:		; preds = %reserved_word
	ret i32 undef

bb1930:		; preds = %reserved_word
	ret i32 undef

bb1936:		; preds = %reserved_word
	br i1 undef, label %bb2834, label %bb1937

bb1937:		; preds = %bb1936
	ret i32 undef

bb1946:		; preds = %reserved_word
	unreachable

bb1951:		; preds = %reserved_word
	ret i32 undef

bb1962:		; preds = %reserved_word
	ret i32 undef

bb1968:		; preds = %reserved_word
	ret i32 280

bb1969:		; preds = %reserved_word
	ret i32 276

bb1970:		; preds = %reserved_word
	ret i32 277

bb1971:		; preds = %reserved_word
	ret i32 288

bb1982:		; preds = %reserved_word
	br i1 undef, label %bb2834, label %bb1986

bb1986:		; preds = %bb1982
	ret i32 undef

bb2005:		; preds = %reserved_word
	ret i32 undef

bb2006:		; preds = %reserved_word
	ret i32 282

bb2007:		; preds = %reserved_word
	ret i32 282

bb2008:		; preds = %reserved_word
	ret i32 282

bb2009:		; preds = %reserved_word
	ret i32 282

bb2010:		; preds = %reserved_word
	ret i32 282

bb2011:		; preds = %reserved_word
	ret i32 282

bb2012:		; preds = %reserved_word
	unreachable

bb2079:		; preds = %reserved_word
	ret i32 undef

bb2080:		; preds = %reserved_word
	ret i32 282

bb2081:		; preds = %reserved_word
	ret i32 undef

bb2087:		; preds = %reserved_word
	ret i32 undef

bb2088:		; preds = %reserved_word
	ret i32 287

bb2089:		; preds = %reserved_word
	ret i32 287

bb2090:		; preds = %reserved_word
	ret i32 undef

bb2091:		; preds = %reserved_word
	ret i32 280

bb2102:		; preds = %reserved_word
	ret i32 282

bb2108:		; preds = %reserved_word
	ret i32 undef

bb2114:		; preds = %reserved_word
	ret i32 undef

bb2115:		; preds = %reserved_word
	ret i32 282

bb2116:		; preds = %reserved_word
	ret i32 282

bb2137:		; preds = %reserved_word
	ret i32 undef

bb2138:		; preds = %reserved_word
	ret i32 282

bb2144:		; preds = %reserved_word
	ret i32 undef

bb2145:		; preds = %reserved_word
	ret i32 282

bb2146:		; preds = %reserved_word
	ret i32 undef

bb2147:		; preds = %reserved_word
	ret i32 undef

bb2148:		; preds = %reserved_word
	ret i32 282

bb2154:		; preds = %reserved_word
	ret i32 undef

bb2155:		; preds = %reserved_word
	ret i32 282

bb2166:		; preds = %reserved_word
	ret i32 282

bb2167:		; preds = %reserved_word
	ret i32 undef

bb2173:		; preds = %reserved_word
	ret i32 274

bb2174:		; preds = %reserved_word
	ret i32 undef

bb2175:		; preds = %reserved_word
	br i1 undef, label %bb2834, label %bb2176

bb2176:		; preds = %bb2175
	ret i32 undef

bb2180:		; preds = %reserved_word
	ret i32 undef

bb2181:		; preds = %reserved_word
	ret i32 undef

bb2187:		; preds = %reserved_word
	ret i32 undef

bb2188:		; preds = %reserved_word
	ret i32 280

bb2199:		; preds = %reserved_word
	ret i32 295

bb2205:		; preds = %reserved_word
	ret i32 287

bb2206:		; preds = %reserved_word
	ret i32 287

bb2217:		; preds = %reserved_word
	ret i32 undef

bb2218:		; preds = %reserved_word
	ret i32 undef

bb2229:		; preds = %reserved_word
	unreachable

bb2233:		; preds = %reserved_word
	ret i32 undef

bb2234:		; preds = %reserved_word
	ret i32 undef

bb2235:		; preds = %reserved_word
	ret i32 undef

bb2236:		; preds = %reserved_word
	ret i32 undef

bb2237:		; preds = %reserved_word
	ret i32 undef

bb2238:		; preds = %reserved_word
	ret i32 undef

bb2239:		; preds = %reserved_word, %reserved_word
	unreachable

bb2267:		; preds = %reserved_word
	ret i32 280

bb2268:		; preds = %reserved_word
	ret i32 288

bb2276:		; preds = %reserved_word
	unreachable

bb2337:		; preds = %reserved_word
	ret i32 300

bb2348:		; preds = %reserved_word
	ret i32 undef

bb2349:		; preds = %reserved_word
	ret i32 undef

bb2350:		; preds = %reserved_word
	ret i32 undef

bb2356:		; preds = %reserved_word
	ret i32 undef

bb2357:		; preds = %reserved_word
	br i1 undef, label %bb2834, label %bb2358

bb2358:		; preds = %bb2357
	ret i32 undef

bb2367:		; preds = %reserved_word
	ret i32 undef

bb2368:		; preds = %reserved_word
	ret i32 270

bb2369:		; preds = %reserved_word
	ret i32 undef

bb2370:		; preds = %reserved_word
	unreachable

bb2381:		; preds = %reserved_word
	unreachable

bb2445:		; preds = %reserved_word
	unreachable

bb2453:		; preds = %reserved_word
	unreachable

bb2457:		; preds = %reserved_word
	unreachable

bb2463:		; preds = %reserved_word
	ret i32 286

bb2464:		; preds = %reserved_word
	unreachable

bb2503:		; preds = %reserved_word
	ret i32 280

bb2504:		; preds = %reserved_word
	ret i32 undef

bb2515:		; preds = %reserved_word
	ret i32 undef

bb2516:		; preds = %reserved_word
	ret i32 undef

bb2522:		; preds = %reserved_word
	unreachable

bb2527:		; preds = %reserved_word
	unreachable

bb2537:		; preds = %reserved_word
	ret i32 undef

bb2538:		; preds = %reserved_word
	ret i32 undef

bb2549:		; preds = %reserved_word
	unreachable

bb2555:		; preds = %reserved_word
	br i1 undef, label %bb2834, label %bb2556

bb2556:		; preds = %bb2555
	ret i32 undef

bb2565:		; preds = %reserved_word
	ret i32 undef

bb2566:		; preds = %reserved_word
	ret i32 undef

bb2567:		; preds = %reserved_word
	ret i32 undef

bb2568:		; preds = %reserved_word
	ret i32 undef

bb2569:		; preds = %reserved_word
	ret i32 undef

bb2570:		; preds = %reserved_word
	ret i32 undef

bb2571:		; preds = %reserved_word
	ret i32 undef

bb2572:		; preds = %reserved_word
	ret i32 undef

bb2583:		; preds = %reserved_word
	br i1 undef, label %bb2834, label %bb2584

bb2584:		; preds = %bb2583
	ret i32 undef

bb2593:		; preds = %reserved_word
	ret i32 282

bb2594:		; preds = %reserved_word
	ret i32 282

bb2595:		; preds = %reserved_word
	ret i32 undef

bb2596:		; preds = %reserved_word
	ret i32 undef

bb2602:		; preds = %reserved_word
	ret i32 undef

bb2603:		; preds = %reserved_word
	ret i32 undef

bb2604:		; preds = %reserved_word
	ret i32 undef

bb2605:		; preds = %reserved_word
	ret i32 undef

bb2606:		; preds = %reserved_word
	ret i32 undef

bb2617:		; preds = %reserved_word
	ret i32 undef

bb2618:		; preds = %reserved_word
	ret i32 undef

bb2619:		; preds = %reserved_word
	unreachable

bb2625:		; preds = %reserved_word
	ret i32 undef

bb2626:		; preds = %reserved_word
	ret i32 undef

bb2627:		; preds = %reserved_word
	ret i32 undef

bb2648:		; preds = %reserved_word
	ret i32 undef

really_sub:		; preds = %reserved_word, %reserved_word
	unreachable

bb2737:		; preds = %reserved_word
	ret i32 undef

bb2738:		; preds = %reserved_word
	ret i32 undef

bb2739:		; preds = %reserved_word
	ret i32 undef

bb2740:		; preds = %reserved_word
	ret i32 undef

bb2741:		; preds = %reserved_word
	ret i32 undef

bb2742:		; preds = %reserved_word
	ret i32 undef

bb2743:		; preds = %reserved_word
	ret i32 undef

bb2744:		; preds = %reserved_word
	unreachable

bb2758:		; preds = %reserved_word
	ret i32 undef

bb2764:		; preds = %reserved_word
	ret i32 282

bb2765:		; preds = %reserved_word
	ret i32 282

bb2766:		; preds = %reserved_word
	ret i32 undef

bb2782:		; preds = %reserved_word
	ret i32 273

bb2783:		; preds = %reserved_word
	ret i32 275

bb2784:		; preds = %reserved_word
	ret i32 undef

bb2785:		; preds = %reserved_word
	br i1 undef, label %bb2834, label %bb2786

bb2786:		; preds = %bb2785
	ret i32 undef

bb2790:		; preds = %reserved_word
	ret i32 undef

bb2791:		; preds = %reserved_word
	ret i32 undef

bb2797:		; preds = %reserved_word
	ret i32 undef

bb2815:		; preds = %reserved_word
	ret i32 undef

bb2816:		; preds = %reserved_word
	ret i32 272

bb2817:		; preds = %reserved_word
	ret i32 undef

bb2818:		; preds = %reserved_word
	ret i32 282

bb2819:		; preds = %reserved_word
	ret i32 undef

bb2820:		; preds = %reserved_word
	ret i32 282

bb2821:		; preds = %reserved_word
	unreachable

bb2826:		; preds = %reserved_word
	unreachable

bb2829:		; preds = %reserved_word
	ret i32 300

bb2830:		; preds = %reserved_word
	unreachable

bb2834:		; preds = %bb2785, %bb2583, %bb2555, %bb2357, %bb2175, %bb1982, %bb1936
	ret i32 283
}
