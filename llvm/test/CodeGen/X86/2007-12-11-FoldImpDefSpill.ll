; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin | not grep IMPLICIT_DEF

	%struct.__sbuf = type { i8*, i32 }
	%struct.ggBRDF = type { i32 (...)** }
	%"struct.ggBST<ggMaterial>" = type { %"struct.ggBSTNode<ggMaterial>"*, i32 }
	%"struct.ggBST<ggRasterSurfaceTexture>" = type { %"struct.ggBSTNode<ggRasterSurfaceTexture>"*, i32 }
	%"struct.ggBST<ggSolidTexture>" = type { %"struct.ggBSTNode<ggSolidTexture>"*, i32 }
	%"struct.ggBST<ggSpectrum>" = type { %"struct.ggBSTNode<ggSpectrum>"*, i32 }
	%"struct.ggBST<mrObjectRecord>" = type { %"struct.ggBSTNode<mrObjectRecord>"*, i32 }
	%"struct.ggBSTNode<ggMaterial>" = type { %"struct.ggBSTNode<ggMaterial>"*, %"struct.ggBSTNode<ggMaterial>"*, %struct.ggString, %struct.ggMaterial* }
	%"struct.ggBSTNode<ggRasterSurfaceTexture>" = type { %"struct.ggBSTNode<ggRasterSurfaceTexture>"*, %"struct.ggBSTNode<ggRasterSurfaceTexture>"*, %struct.ggString, %struct.ggRasterSurfaceTexture* }
	%"struct.ggBSTNode<ggSolidTexture>" = type { %"struct.ggBSTNode<ggSolidTexture>"*, %"struct.ggBSTNode<ggSolidTexture>"*, %struct.ggString, %struct.ggBRDF* }
	%"struct.ggBSTNode<ggSpectrum>" = type { %"struct.ggBSTNode<ggSpectrum>"*, %"struct.ggBSTNode<ggSpectrum>"*, %struct.ggString, %struct.ggSpectrum* }
	%"struct.ggBSTNode<mrObjectRecord>" = type { %"struct.ggBSTNode<mrObjectRecord>"*, %"struct.ggBSTNode<mrObjectRecord>"*, %struct.ggString, %struct.mrObjectRecord* }
	%"struct.ggDictionary<ggMaterial>" = type { %"struct.ggBST<ggMaterial>" }
	%"struct.ggDictionary<ggRasterSurfaceTexture>" = type { %"struct.ggBST<ggRasterSurfaceTexture>" }
	%"struct.ggDictionary<ggSolidTexture>" = type { %"struct.ggBST<ggSolidTexture>" }
	%"struct.ggDictionary<ggSpectrum>" = type { %"struct.ggBST<ggSpectrum>" }
	%"struct.ggDictionary<mrObjectRecord>" = type { %"struct.ggBST<mrObjectRecord>" }
	%struct.ggHAffineMatrix3 = type { %struct.ggHMatrix3 }
	%struct.ggHBoxMatrix3 = type { %struct.ggHAffineMatrix3 }
	%struct.ggHMatrix3 = type { [4 x [4 x double]] }
	%struct.ggMaterial = type { i32 (...)**, %struct.ggBRDF* }
	%struct.ggPoint3 = type { [3 x double] }
	%"struct.ggRGBPixel<char>" = type { [3 x i8], i8 }
	%"struct.ggRaster<ggRGBPixel<unsigned char> >" = type { i32, i32, %"struct.ggRGBPixel<char>"* }
	%struct.ggRasterSurfaceTexture = type { %"struct.ggRaster<ggRGBPixel<unsigned char> >"* }
	%struct.ggSolidNoise3 = type { i32, [256 x %struct.ggPoint3], [256 x i32] }
	%struct.ggSpectrum = type { [8 x float] }
	%struct.ggString = type { %"struct.ggString::StringRep"* }
	%"struct.ggString::StringRep" = type { i32, i32, [1 x i8] }
	%"struct.ggTrain<mrPixelRenderer*>" = type { %struct.ggBRDF**, i32, i32 }
	%struct.mrObjectRecord = type { %struct.ggHBoxMatrix3, %struct.ggHBoxMatrix3, %struct.mrSurfaceList, %struct.ggMaterial*, i32, %struct.ggRasterSurfaceTexture*, %struct.ggBRDF*, i32, i32 }
	%struct.mrScene = type { %struct.ggSpectrum, %struct.ggSpectrum, %struct.ggBRDF*, %struct.ggBRDF*, %struct.ggBRDF*, i32, double, %"struct.ggDictionary<mrObjectRecord>", %"struct.ggDictionary<ggRasterSurfaceTexture>", %"struct.ggDictionary<ggSolidTexture>", %"struct.ggDictionary<ggSpectrum>", %"struct.ggDictionary<ggMaterial>" }
	%struct.mrSurfaceList = type { %struct.ggBRDF, %"struct.ggTrain<mrPixelRenderer*>" }
	%"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>" = type { %"struct.std::locale::facet" }
	%"struct.std::basic_ios<char,std::char_traits<char> >" = type { %"struct.std::ios_base", %"struct.std::basic_ostream<char,std::char_traits<char> >"*, i8, i8, %"struct.std::basic_streambuf<char,std::char_traits<char> >"*, %"struct.std::ctype<char>"*, %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"*, %"struct.std::__codecvt_abstract_base<char,char,__mbstate_t>"* }
	%"struct.std::basic_istream<char,std::char_traits<char> >" = type { i32 (...)**, i32, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_ostream<char,std::char_traits<char> >" = type { i32 (...)**, %"struct.std::basic_ios<char,std::char_traits<char> >" }
	%"struct.std::basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"struct.std::locale" }
	%"struct.std::ctype<char>" = type { %"struct.std::locale::facet", i32*, i8, i32*, i32*, i32*, i8, [256 x i8], [256 x i8], i8 }
	%"struct.std::ios_base" = type { i32 (...)**, i32, i32, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %struct.__sbuf, [8 x %struct.__sbuf], i32, %struct.__sbuf*, %"struct.std::locale" }
	%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"struct.std::ios_base"*, i32)*, i32, i32 }
	%"struct.std::locale" = type { %"struct.std::locale::_Impl"* }
	%"struct.std::locale::_Impl" = type { i32, %"struct.std::locale::facet"**, i32, %"struct.std::locale::facet"**, i8** }
	%"struct.std::locale::facet" = type { i32 (...)**, i32 }
@.str80 = external constant [7 x i8]		; <[7 x i8]*> [#uses=1]
@.str81 = external constant [11 x i8]		; <[11 x i8]*> [#uses=1]

define fastcc void @_ZN7mrScene4ReadERSi(%struct.mrScene* %this, %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces) {
entry:
	%tmp6.i.i8288 = invoke i8* @_Znam( i32 12 )
			to label %_ZN8ggStringC1Ei.exit unwind label %lpad		; <i8*> [#uses=0]

_ZN8ggStringC1Ei.exit:		; preds = %entry
	%tmp6.i.i8995 = invoke i8* @_Znam( i32 12 )
			to label %_ZN8ggStringC1Ei.exit96 unwind label %lpad3825		; <i8*> [#uses=0]

_ZN8ggStringC1Ei.exit96:		; preds = %_ZN8ggStringC1Ei.exit
	%tmp6.i.i97103 = invoke i8* @_Znam( i32 12 )
			to label %_ZN8ggStringC1Ei.exit104 unwind label %lpad3829		; <i8*> [#uses=0]

_ZN8ggStringC1Ei.exit104:		; preds = %_ZN8ggStringC1Ei.exit96
	%tmp6.i.i105111 = invoke i8* @_Znam( i32 12 )
			to label %_ZN8ggStringC1Ei.exit112 unwind label %lpad3833		; <i8*> [#uses=0]

_ZN8ggStringC1Ei.exit112:		; preds = %_ZN8ggStringC1Ei.exit104
	%tmp6.i.i122128 = invoke i8* @_Znam( i32 12 )
			to label %_ZN8ggStringC1Ei.exit129 unwind label %lpad3837		; <i8*> [#uses=0]

_ZN8ggStringC1Ei.exit129:		; preds = %_ZN8ggStringC1Ei.exit112
	%tmp6.i.i132138 = invoke i8* @_Znam( i32 12 )
			to label %_ZN8ggStringC1Ei.exit139 unwind label %lpad3841		; <i8*> [#uses=0]

_ZN8ggStringC1Ei.exit139:		; preds = %_ZN8ggStringC1Ei.exit129
	%tmp295 = invoke i8* @_Znwm( i32 16 )
			to label %invcont294 unwind label %lpad3845		; <i8*> [#uses=0]

invcont294:		; preds = %_ZN8ggStringC1Ei.exit139
	%tmp10.i.i141 = invoke i8* @_Znam( i32 16 )
			to label %_ZN13mrSurfaceListC1Ev.exit unwind label %lpad3849		; <i8*> [#uses=0]

_ZN13mrSurfaceListC1Ev.exit:		; preds = %invcont294
	%tmp3.i148 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i.noexc:		; preds = %_ZN13mrSurfaceListC1Ev.exit
	%tmp15.i149 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i.noexc unwind label %lpad3845		; <i8*> [#uses=0]

tmp15.i.noexc:		; preds = %tmp3.i.noexc
	br i1 false, label %bb308, label %bb.i

bb.i:		; preds = %tmp15.i.noexc
	ret void

bb308:		; preds = %tmp15.i.noexc
	br i1 false, label %bb3743.preheader, label %bb315

bb3743.preheader:		; preds = %bb308
	%tmp16.i3862 = getelementptr %struct.ggPoint3* null, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	%tmp16.i3859 = getelementptr %struct.ggPoint3* null, i32 0, i32 0, i32 0		; <double*> [#uses=3]
	br label %bb3743

bb315:		; preds = %bb308
	ret void

bb333:		; preds = %invcont3758, %invcont335
	%tmp3.i167180 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i167.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i167.noexc:		; preds = %bb333
	%tmp15.i182 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i.noexc181 unwind label %lpad3845		; <i8*> [#uses=0]

tmp15.i.noexc181:		; preds = %tmp3.i167.noexc
	br i1 false, label %invcont335, label %bb.i178

bb.i178:		; preds = %tmp15.i.noexc181
	ret void

invcont335:		; preds = %tmp15.i.noexc181
	br i1 false, label %bb3743, label %bb333

bb345:		; preds = %invcont3758
	br i1 false, label %bb353, label %bb360

bb353:		; preds = %bb345
	%tmp356 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, double* null )
			to label %bb3743 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

bb360:		; preds = %bb345
	br i1 false, label %bb368, label %bb374

bb368:		; preds = %bb360
	%tmp373 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, double* null )
			to label %bb3743 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

bb374:		; preds = %bb360
	br i1 false, label %bb396, label %bb421

bb396:		; preds = %bb374
	ret void

bb421:		; preds = %bb374
	br i1 false, label %bb429, label %bb530

bb429:		; preds = %bb421
	ret void

bb530:		; preds = %bb421
	br i1 false, label %bb538, label %bb673

bb538:		; preds = %bb530
	ret void

bb673:		; preds = %bb530
	br i1 false, label %bb681, label %bb778

bb681:		; preds = %bb673
	ret void

bb778:		; preds = %bb673
	br i1 false, label %bb786, label %bb891

bb786:		; preds = %bb778
	ret void

bb891:		; preds = %bb778
	br i1 false, label %bb899, label %bb998

bb899:		; preds = %bb891
	ret void

bb998:		; preds = %bb891
	br i1 false, label %bb1168, label %bb1190

bb1168:		; preds = %bb998
	ret void

bb1190:		; preds = %bb998
	br i1 false, label %bb1198, label %bb1220

bb1198:		; preds = %bb1190
	ret void

bb1220:		; preds = %bb1190
	br i1 false, label %bb1228, label %bb1250

bb1228:		; preds = %bb1220
	ret void

bb1250:		; preds = %bb1220
	br i1 false, label %bb1258, label %bb1303

bb1258:		; preds = %bb1250
	ret void

bb1303:		; preds = %bb1250
	br i1 false, label %bb1311, label %bb1366

bb1311:		; preds = %bb1303
	ret void

bb1366:		; preds = %bb1303
	br i1 false, label %bb1374, label %bb1432

bb1374:		; preds = %bb1366
	ret void

bb1432:		; preds = %bb1366
	br i1 false, label %bb1440, label %bb1495

bb1440:		; preds = %bb1432
	ret void

bb1495:		; preds = %bb1432
	br i1 false, label %bb1503, label %bb1561

bb1503:		; preds = %bb1495
	ret void

bb1561:		; preds = %bb1495
	br i1 false, label %bb1569, label %bb1624

bb1569:		; preds = %bb1561
	ret void

bb1624:		; preds = %bb1561
	br i1 false, label %bb1632, label %bb1654

bb1632:		; preds = %bb1624
	store double 0.000000e+00, double* %tmp16.i3859, align 8
	%tmp3.i38383852 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i3838.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i3838.noexc:		; preds = %bb1632
	%tmp15.i38473853 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i3847.noexc unwind label %lpad3845		; <i8*> [#uses=0]

tmp15.i3847.noexc:		; preds = %tmp3.i3838.noexc
	br i1 false, label %invcont1634, label %bb.i3850

bb.i3850:		; preds = %tmp15.i3847.noexc
	ret void

invcont1634:		; preds = %tmp15.i3847.noexc
	%tmp3.i38173831 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i3817.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i3817.noexc:		; preds = %invcont1634
	%tmp15.i38263832 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i3826.noexc unwind label %lpad3845		; <i8*> [#uses=0]

tmp15.i3826.noexc:		; preds = %tmp3.i3817.noexc
	br i1 false, label %invcont1636, label %bb.i3829

bb.i3829:		; preds = %tmp15.i3826.noexc
	ret void

invcont1636:		; preds = %tmp15.i3826.noexc
	%tmp8.i38083811 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, double* %tmp16.i3862 )
			to label %tmp8.i3808.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

tmp8.i3808.noexc:		; preds = %invcont1636
	%tmp9.i38093812 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp8.i38083811, double* null )
			to label %tmp9.i3809.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

tmp9.i3809.noexc:		; preds = %tmp8.i3808.noexc
	%tmp10.i38103813 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp9.i38093812, double* null )
			to label %invcont1638 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

invcont1638:		; preds = %tmp9.i3809.noexc
	%tmp8.i37983801 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, double* %tmp16.i3859 )
			to label %tmp8.i3798.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

tmp8.i3798.noexc:		; preds = %invcont1638
	%tmp9.i37993802 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp8.i37983801, double* null )
			to label %tmp9.i3799.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

tmp9.i3799.noexc:		; preds = %tmp8.i3798.noexc
	%tmp10.i38003803 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp9.i37993802, double* null )
			to label %invcont1640 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

invcont1640:		; preds = %tmp9.i3799.noexc
	%tmp3.i3778 = load double* %tmp16.i3859, align 8		; <double> [#uses=1]
	%tmp1643 = invoke i8* @_Znwm( i32 76 )
			to label %invcont1642 unwind label %lpad3845		; <i8*> [#uses=0]

invcont1642:		; preds = %invcont1640
	%tmp18.i3770 = sub double %tmp3.i3778, 0.000000e+00		; <double> [#uses=0]
	invoke fastcc void @_ZN7mrScene9AddObjectEP9mrSurfaceRK8ggStringS4_i( %struct.mrScene* %this, %struct.ggBRDF* null, %struct.ggString* null, %struct.ggString* null, i32 0 )
			to label %bb3743 unwind label %lpad3845

bb1654:		; preds = %bb1624
	br i1 false, label %bb1662, label %bb1693

bb1662:		; preds = %bb1654
	%tmp3.i37143728 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i3714.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i3714.noexc:		; preds = %bb1662
	%tmp15.i37233729 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i3723.noexc unwind label %lpad3845		; <i8*> [#uses=0]

tmp15.i3723.noexc:		; preds = %tmp3.i3714.noexc
	ret void

bb1693:		; preds = %bb1654
	br i1 false, label %bb1701, label %bb1745

bb1701:		; preds = %bb1693
	%tmp3.i36493663 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i3649.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i3649.noexc:		; preds = %bb1701
	ret void

bb1745:		; preds = %bb1693
	br i1 false, label %bb1753, label %bb1797

bb1753:		; preds = %bb1745
	ret void

bb1797:		; preds = %bb1745
	br i1 false, label %bb1805, label %bb1847

bb1805:		; preds = %bb1797
	ret void

bb1847:		; preds = %bb1797
	br i1 false, label %bb1855, label %bb1897

bb1855:		; preds = %bb1847
	%tmp3.i34633477 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i3463.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i3463.noexc:		; preds = %bb1855
	%tmp15.i34723478 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i3472.noexc unwind label %lpad3845		; <i8*> [#uses=0]

tmp15.i3472.noexc:		; preds = %tmp3.i3463.noexc
	br i1 false, label %invcont1857, label %bb.i3475

bb.i3475:		; preds = %tmp15.i3472.noexc
	invoke fastcc void @_ZN8ggStringaSEPKc( %struct.ggString* null, i8* null )
			to label %invcont1857 unwind label %lpad3845

invcont1857:		; preds = %bb.i3475, %tmp15.i3472.noexc
	%tmp1860 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, double* null )
			to label %invcont1859 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

invcont1859:		; preds = %invcont1857
	%tmp1862 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp1860, double* null )
			to label %invcont1861 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

invcont1861:		; preds = %invcont1859
	%tmp1864 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp1862, double* null )
			to label %invcont1863 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

invcont1863:		; preds = %invcont1861
	%tmp1866 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp1864, double* null )
			to label %invcont1865 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

invcont1865:		; preds = %invcont1863
	%tmp1868 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp1866, double* null )
			to label %invcont1867 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

invcont1867:		; preds = %invcont1865
	%tmp1881 = invoke i8 @_ZNKSt9basic_iosIcSt11char_traitsIcEE4goodEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null ) zeroext 
			to label %invcont1880 unwind label %lpad3845		; <i8> [#uses=0]

invcont1880:		; preds = %invcont1867
	%tmp1883 = invoke i8* @_Znwm( i32 24 )
			to label %invcont1882 unwind label %lpad3845		; <i8*> [#uses=0]

invcont1882:		; preds = %invcont1880
	invoke fastcc void @_ZN7mrScene9AddObjectEP9mrSurfaceRK8ggStringS4_i( %struct.mrScene* %this, %struct.ggBRDF* null, %struct.ggString* null, %struct.ggString* null, i32 0 )
			to label %bb3743 unwind label %lpad3845

bb1897:		; preds = %bb1847
	br i1 false, label %bb1905, label %bb1947

bb1905:		; preds = %bb1897
	ret void

bb1947:		; preds = %bb1897
	br i1 false, label %bb1955, label %bb2000

bb1955:		; preds = %bb1947
	ret void

bb2000:		; preds = %bb1947
	br i1 false, label %bb2008, label %bb2053

bb2008:		; preds = %bb2000
	ret void

bb2053:		; preds = %bb2000
	br i1 false, label %bb2061, label %bb2106

bb2061:		; preds = %bb2053
	%tmp3.i32433257 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i3243.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i3243.noexc:		; preds = %bb2061
	%tmp15.i32523258 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %bb.i3255 unwind label %lpad3845		; <i8*> [#uses=0]

bb.i3255:		; preds = %tmp3.i3243.noexc
	invoke fastcc void @_ZN8ggStringaSEPKc( %struct.ggString* null, i8* null )
			to label %invcont2063 unwind label %lpad3845

invcont2063:		; preds = %bb.i3255
	ret void

bb2106:		; preds = %bb2053
	%tmp7.i3214 = call i32 @strcmp( i8* %tmp5.i161, i8* getelementptr ([7 x i8]* @.str80, i32 0, i32 0) ) nounwind readonly 		; <i32> [#uses=0]
	br i1 false, label %bb2114, label %bb2136

bb2114:		; preds = %bb2106
	%tmp3.i31923206 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i3192.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i3192.noexc:		; preds = %bb2114
	%tmp15.i32013207 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i3201.noexc unwind label %lpad3845		; <i8*> [#uses=0]

tmp15.i3201.noexc:		; preds = %tmp3.i3192.noexc
	br i1 false, label %invcont2116, label %bb.i3204

bb.i3204:		; preds = %tmp15.i3201.noexc
	ret void

invcont2116:		; preds = %tmp15.i3201.noexc
	%tmp3.i31713185 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i3171.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i3171.noexc:		; preds = %invcont2116
	%tmp15.i31803186 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i3180.noexc unwind label %lpad3845		; <i8*> [#uses=0]

tmp15.i3180.noexc:		; preds = %tmp3.i3171.noexc
	br i1 false, label %invcont2118, label %bb.i3183

bb.i3183:		; preds = %tmp15.i3180.noexc
	ret void

invcont2118:		; preds = %tmp15.i3180.noexc
	%tmp8.i31623165 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, double* null )
			to label %tmp8.i3162.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

tmp8.i3162.noexc:		; preds = %invcont2118
	%tmp9.i31633166 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp8.i31623165, double* null )
			to label %tmp9.i3163.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

tmp9.i3163.noexc:		; preds = %tmp8.i3162.noexc
	%tmp10.i31643167 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp9.i31633166, double* null )
			to label %invcont2120 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

invcont2120:		; preds = %tmp9.i3163.noexc
	%tmp2123 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, double* null )
			to label %invcont2122 unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

invcont2122:		; preds = %invcont2120
	%tmp2125 = invoke i8* @_Znwm( i32 36 )
			to label %invcont2124 unwind label %lpad3845		; <i8*> [#uses=0]

invcont2124:		; preds = %invcont2122
	invoke fastcc void @_ZN7mrScene9AddObjectEP9mrSurfaceRK8ggStringS4_i( %struct.mrScene* %this, %struct.ggBRDF* null, %struct.ggString* null, %struct.ggString* null, i32 0 )
			to label %bb3743 unwind label %lpad3845

bb2136:		; preds = %bb2106
	%tmp7.i3128 = call i32 @strcmp( i8* %tmp5.i161, i8* getelementptr ([11 x i8]* @.str81, i32 0, i32 0) ) nounwind readonly 		; <i32> [#uses=0]
	br i1 false, label %bb2144, label %bb3336

bb2144:		; preds = %bb2136
	%tmp6.i.i31173123 = invoke i8* @_Znam( i32 12 )
			to label %_ZN8ggStringC1Ei.exit3124 unwind label %lpad3845		; <i8*> [#uses=0]

_ZN8ggStringC1Ei.exit3124:		; preds = %bb2144
	%tmp3.i30983112 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i3098.noexc unwind label %lpad3921		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i3098.noexc:		; preds = %_ZN8ggStringC1Ei.exit3124
	%tmp15.i31073113 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i3107.noexc unwind label %lpad3921		; <i8*> [#uses=0]

tmp15.i3107.noexc:		; preds = %tmp3.i3098.noexc
	br i1 false, label %invcont2147, label %bb.i3110

bb.i3110:		; preds = %tmp15.i3107.noexc
	ret void

invcont2147:		; preds = %tmp15.i3107.noexc
	%tmp2161 = invoke i8 @_ZNKSt9basic_iosIcSt11char_traitsIcEE4goodEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null ) zeroext 
			to label %invcont2160 unwind label %lpad3921		; <i8> [#uses=0]

invcont2160:		; preds = %invcont2147
	%tmp4.i30933094 = invoke fastcc %struct.ggSpectrum* @_ZN5ggBSTI10ggSpectrumE4findERK8ggString3( %"struct.ggBSTNode<ggSpectrum>"* null, %struct.ggString* null )
			to label %invcont2164 unwind label %lpad3921		; <%struct.ggSpectrum*> [#uses=0]

invcont2164:		; preds = %invcont2160
	br i1 false, label %bb2170, label %bb2181

bb2170:		; preds = %invcont2164
	ret void

bb2181:		; preds = %invcont2164
	invoke fastcc void @_ZN8ggStringD1Ev( %struct.ggString* null )
			to label %bb3743 unwind label %lpad3845

bb3336:		; preds = %bb2136
	br i1 false, label %bb3344, label %bb3734

bb3344:		; preds = %bb3336
	%tmp6.i.i773779 = invoke i8* @_Znam( i32 12 )
			to label %_ZN8ggStringC1Ei.exit780 unwind label %lpad3845		; <i8*> [#uses=0]

_ZN8ggStringC1Ei.exit780:		; preds = %bb3344
	%tmp6.i.i765771 = invoke i8* @_Znam( i32 12 )
			to label %_ZN8ggStringC1Ei.exit772 unwind label %lpad4025		; <i8*> [#uses=0]

_ZN8ggStringC1Ei.exit772:		; preds = %_ZN8ggStringC1Ei.exit780
	%tmp3.i746760 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i746.noexc unwind label %lpad4029		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i746.noexc:		; preds = %_ZN8ggStringC1Ei.exit772
	%tmp15.i755761 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i755.noexc unwind label %lpad4029		; <i8*> [#uses=0]

tmp15.i755.noexc:		; preds = %tmp3.i746.noexc
	br i1 false, label %invcont3348, label %bb.i758

bb.i758:		; preds = %tmp15.i755.noexc
	ret void

invcont3348:		; preds = %tmp15.i755.noexc
	%tmp3.i726740 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i726.noexc unwind label %lpad4029		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i726.noexc:		; preds = %invcont3348
	%tmp15.i735741 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i735.noexc unwind label %lpad4029		; <i8*> [#uses=0]

tmp15.i735.noexc:		; preds = %tmp3.i726.noexc
	br i1 false, label %bb3458, label %bb.i738

bb.i738:		; preds = %tmp15.i735.noexc
	ret void

bb3458:		; preds = %tmp15.i735.noexc
	br i1 false, label %bb3466, label %bb3491

bb3466:		; preds = %bb3458
	%tmp3469 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, double* null )
			to label %invcont3468 unwind label %lpad4029		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

invcont3468:		; preds = %bb3466
	%tmp3471 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp3469, double* null )
			to label %invcont3470 unwind label %lpad4029		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=1]

invcont3470:		; preds = %invcont3468
	%tmp3473 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERi( %"struct.std::basic_istream<char,std::char_traits<char> >"* %tmp3471, i32* null )
			to label %invcont3472 unwind label %lpad4029		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

invcont3472:		; preds = %invcont3470
	%tmp3475 = invoke i8* @_Znwm( i32 7196 )
			to label %invcont3474 unwind label %lpad4029		; <i8*> [#uses=1]

invcont3474:		; preds = %invcont3472
	invoke fastcc void @_ZN13ggSolidNoise3C1Ev( %struct.ggSolidNoise3* null )
			to label %_ZN22ggCoverageSolidTextureC1Eddi.exit unwind label %lpad4045

_ZN22ggCoverageSolidTextureC1Eddi.exit:		; preds = %invcont3474
	%tmp34823483 = bitcast i8* %tmp3475 to %struct.ggBRDF*		; <%struct.ggBRDF*> [#uses=2]
	invoke fastcc void @_ZN5ggBSTI14ggSolidTextureE17InsertIntoSubtreeERK8ggStringPS0_RP9ggBSTNodeIS0_E( %"struct.ggBST<ggSolidTexture>"* null, %struct.ggString* null, %struct.ggBRDF* %tmp34823483, %"struct.ggBSTNode<ggSolidTexture>"** null )
			to label %bb3662 unwind label %lpad4029

bb3491:		; preds = %bb3458
	ret void

bb3662:		; preds = %_ZN22ggCoverageSolidTextureC1Eddi.exit
	invoke fastcc void @_ZN8ggStringD1Ev( %struct.ggString* null )
			to label %invcont3663 unwind label %lpad4025

invcont3663:		; preds = %bb3662
	invoke fastcc void @_ZN8ggStringD1Ev( %struct.ggString* null )
			to label %bb3743 unwind label %lpad3845

bb3734:		; preds = %bb3336
	ret void

bb3743:		; preds = %invcont3663, %bb2181, %invcont2124, %invcont1882, %invcont1642, %bb368, %bb353, %invcont335, %bb3743.preheader
	%tex1.3 = phi %struct.ggBRDF* [ undef, %bb3743.preheader ], [ %tex1.3, %bb368 ], [ %tex1.3, %invcont1642 ], [ %tex1.3, %invcont1882 ], [ %tex1.3, %invcont2124 ], [ %tex1.3, %bb2181 ], [ %tex1.3, %invcont335 ], [ %tmp34823483, %invcont3663 ], [ %tex1.3, %bb353 ]		; <%struct.ggBRDF*> [#uses=7]
	%tmp3.i312325 = invoke %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_( %"struct.std::basic_istream<char,std::char_traits<char> >"* %surfaces, i8* null )
			to label %tmp3.i312.noexc unwind label %lpad3845		; <%"struct.std::basic_istream<char,std::char_traits<char> >"*> [#uses=0]

tmp3.i312.noexc:		; preds = %bb3743
	%tmp15.i327 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %tmp15.i.noexc326 unwind label %lpad3845		; <i8*> [#uses=0]

tmp15.i.noexc326:		; preds = %tmp3.i312.noexc
	br i1 false, label %invcont3745, label %bb.i323

bb.i323:		; preds = %tmp15.i.noexc326
	ret void

invcont3745:		; preds = %tmp15.i.noexc326
	%tmp3759 = invoke i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv( %"struct.std::basic_ios<char,std::char_traits<char> >"* null )
			to label %invcont3758 unwind label %lpad3845		; <i8*> [#uses=0]

invcont3758:		; preds = %invcont3745
	%tmp5.i161 = getelementptr %"struct.ggString::StringRep"* null, i32 0, i32 2, i32 0		; <i8*> [#uses=2]
	br i1 false, label %bb333, label %bb345

lpad:		; preds = %entry
	ret void

lpad3825:		; preds = %_ZN8ggStringC1Ei.exit
	ret void

lpad3829:		; preds = %_ZN8ggStringC1Ei.exit96
	ret void

lpad3833:		; preds = %_ZN8ggStringC1Ei.exit104
	ret void

lpad3837:		; preds = %_ZN8ggStringC1Ei.exit112
	ret void

lpad3841:		; preds = %_ZN8ggStringC1Ei.exit129
	ret void

lpad3845:		; preds = %invcont3745, %tmp3.i312.noexc, %bb3743, %invcont3663, %bb3344, %bb2181, %bb2144, %invcont2124, %invcont2122, %invcont2120, %tmp9.i3163.noexc, %tmp8.i3162.noexc, %invcont2118, %tmp3.i3171.noexc, %invcont2116, %tmp3.i3192.noexc, %bb2114, %bb.i3255, %tmp3.i3243.noexc, %bb2061, %invcont1882, %invcont1880, %invcont1867, %invcont1865, %invcont1863, %invcont1861, %invcont1859, %invcont1857, %bb.i3475, %tmp3.i3463.noexc, %bb1855, %bb1701, %tmp3.i3714.noexc, %bb1662, %invcont1642, %invcont1640, %tmp9.i3799.noexc, %tmp8.i3798.noexc, %invcont1638, %tmp9.i3809.noexc, %tmp8.i3808.noexc, %invcont1636, %tmp3.i3817.noexc, %invcont1634, %tmp3.i3838.noexc, %bb1632, %bb368, %bb353, %tmp3.i167.noexc, %bb333, %tmp3.i.noexc, %_ZN13mrSurfaceListC1Ev.exit, %_ZN8ggStringC1Ei.exit139
	ret void

lpad3849:		; preds = %invcont294
	ret void

lpad3921:		; preds = %invcont2160, %invcont2147, %tmp3.i3098.noexc, %_ZN8ggStringC1Ei.exit3124
	ret void

lpad4025:		; preds = %bb3662, %_ZN8ggStringC1Ei.exit780
	ret void

lpad4029:		; preds = %_ZN22ggCoverageSolidTextureC1Eddi.exit, %invcont3472, %invcont3470, %invcont3468, %bb3466, %tmp3.i726.noexc, %invcont3348, %tmp3.i746.noexc, %_ZN8ggStringC1Ei.exit772
	ret void

lpad4045:		; preds = %invcont3474
	ret void
}

declare fastcc void @_ZN8ggStringD1Ev(%struct.ggString*)

declare i8* @_Znam(i32)

declare fastcc void @_ZN8ggStringaSEPKc(%struct.ggString*, i8*)

declare i32 @strcmp(i8*, i8*) nounwind readonly 

declare %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERi(%"struct.std::basic_istream<char,std::char_traits<char> >"*, i32*)

declare i8* @_Znwm(i32)

declare i8* @_ZNKSt9basic_iosIcSt11char_traitsIcEEcvPvEv(%"struct.std::basic_ios<char,std::char_traits<char> >"*)

declare %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZNSirsERd(%"struct.std::basic_istream<char,std::char_traits<char> >"*, double*)

declare %"struct.std::basic_istream<char,std::char_traits<char> >"* @_ZStrsIcSt11char_traitsIcEERSt13basic_istreamIT_T0_ES6_PS3_(%"struct.std::basic_istream<char,std::char_traits<char> >"*, i8*)

declare fastcc void @_ZN13ggSolidNoise3C1Ev(%struct.ggSolidNoise3*)

declare i8 @_ZNKSt9basic_iosIcSt11char_traitsIcEE4goodEv(%"struct.std::basic_ios<char,std::char_traits<char> >"*) zeroext 

declare fastcc %struct.ggSpectrum* @_ZN5ggBSTI10ggSpectrumE4findERK8ggString3(%"struct.ggBSTNode<ggSpectrum>"*, %struct.ggString*)

declare fastcc void @_ZN5ggBSTI14ggSolidTextureE17InsertIntoSubtreeERK8ggStringPS0_RP9ggBSTNodeIS0_E(%"struct.ggBST<ggSolidTexture>"*, %struct.ggString*, %struct.ggBRDF*, %"struct.ggBSTNode<ggSolidTexture>"**)

declare fastcc void @_ZN7mrScene9AddObjectEP9mrSurfaceRK8ggStringS4_i(%struct.mrScene*, %struct.ggBRDF*, %struct.ggString*, %struct.ggString*, i32)
