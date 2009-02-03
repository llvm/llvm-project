; RUN: llvm-as < %s | opt -condprop -disable-output
; PR877
target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin9.0.0d1"
	%"struct.kc::impl_Ccode_option" = type { %"struct.kc::impl_abstract_phylum" }
	%"struct.kc::impl_ID" = type { %"struct.kc::impl_abstract_phylum", %"struct.kc::impl_Ccode_option"*, %"struct.kc::impl_casestring__Str"*, i32, %"struct.kc::impl_casestring__Str"* }
	%"struct.kc::impl_abstract_phylum" = type { i32 (...)** }
	%"struct.kc::impl_casestring__Str" = type { %"struct.kc::impl_abstract_phylum", i8* }
	%"struct.kc::impl_elem_patternrepresentation" = type { %"struct.kc::impl_abstract_phylum", i32, %"struct.kc::impl_casestring__Str"*, %"struct.kc::impl_ID"* }
	%"struct.kc::impl_outmostpatterns" = type { %"struct.kc::impl_Ccode_option", %"struct.kc::impl_elem_patternrepresentation"*, %"struct.kc::impl_outmostpatterns"* }
	%"struct.kc::impl_patternrepresentations" = type { %"struct.kc::impl_Ccode_option", %"struct.kc::impl_outmostpatterns"*, %"struct.kc::impl_patternrepresentations"* }

define void @_ZN2kc16compare_patternsEPNS_26impl_patternrepresentationES1_PNS_27impl_patternrepresentationsE() {
entry:
	br label %bb1269.outer.outer.outer.outer

cond_true:		; preds = %cond_true1298
	br label %bb1269.outer69

cond_false:		; preds = %cond_true1298
	br i1 false, label %cond_next, label %bb51

cond_next:		; preds = %cond_false
	br i1 false, label %bb52, label %bb51

bb51:		; preds = %cond_next, %cond_false
	br label %bb52

bb52:		; preds = %bb51, %cond_next
	br i1 false, label %cond_false82, label %cond_true55

cond_true55:		; preds = %bb52
	br i1 false, label %UnifiedReturnBlock, label %cond_true57

cond_true57:		; preds = %cond_true55
	br label %UnifiedReturnBlock

cond_false82:		; preds = %bb52
	br i1 false, label %cond_next97, label %bb113

cond_next97:		; preds = %cond_false82
	br i1 false, label %bb114, label %bb113

bb113:		; preds = %cond_next97, %cond_false82
	br label %bb114

bb114:		; preds = %bb113, %cond_next97
	br i1 false, label %cond_false151, label %cond_true117

cond_true117:		; preds = %bb114
	br i1 false, label %UnifiedReturnBlock, label %cond_true120

cond_true120:		; preds = %cond_true117
	br label %UnifiedReturnBlock

cond_false151:		; preds = %bb114
	br i1 false, label %cond_next166, label %bb182

cond_next166:		; preds = %cond_false151
	br i1 false, label %bb183, label %bb182

bb182:		; preds = %cond_next166, %cond_false151
	br label %bb183

bb183:		; preds = %bb182, %cond_next166
	br i1 false, label %cond_false256, label %cond_true186

cond_true186:		; preds = %bb183
	br i1 false, label %cond_true207, label %cond_false214

cond_true207:		; preds = %cond_true186
	br label %bb1269.outer38.backedge

bb1269.outer38.backedge:		; preds = %cond_true545, %cond_true432, %cond_true320, %cond_true207
	br label %bb1269.outer38

cond_false214:		; preds = %cond_true186
	br i1 false, label %cond_true228, label %cond_false235

cond_true228:		; preds = %cond_false214
	br label %bb1269.outer21.backedge

bb1269.outer21.backedge:		; preds = %cond_true566, %cond_true453, %cond_true341, %cond_true228
	br label %bb1269.outer21

cond_false235:		; preds = %cond_false214
	br i1 false, label %UnifiedReturnBlock, label %cond_false250

cond_false250:		; preds = %cond_false235
	br label %UnifiedUnreachableBlock

cond_false256:		; preds = %bb183
	br i1 false, label %cond_next271, label %bb287

cond_next271:		; preds = %cond_false256
	br i1 false, label %bb288, label %bb287

bb287:		; preds = %cond_next271, %cond_false256
	br label %bb288

bb288:		; preds = %bb287, %cond_next271
	br i1 false, label %cond_false369, label %cond_true291

cond_true291:		; preds = %bb288
	br i1 false, label %cond_true320, label %cond_false327

cond_true320:		; preds = %cond_true291
	br label %bb1269.outer38.backedge

cond_false327:		; preds = %cond_true291
	br i1 false, label %cond_true341, label %cond_false348

cond_true341:		; preds = %cond_false327
	br label %bb1269.outer21.backedge

cond_false348:		; preds = %cond_false327
	br i1 false, label %UnifiedReturnBlock, label %cond_false363

cond_false363:		; preds = %cond_false348
	br label %UnifiedUnreachableBlock

cond_false369:		; preds = %bb288
	br i1 false, label %cond_next384, label %bb400

cond_next384:		; preds = %cond_false369
	br i1 false, label %bb401, label %bb400

bb400:		; preds = %cond_next384, %cond_false369
	br label %bb401

bb401:		; preds = %bb400, %cond_next384
	br i1 false, label %cond_false481, label %cond_true404

cond_true404:		; preds = %bb401
	br i1 false, label %cond_true432, label %cond_false439

cond_true432:		; preds = %cond_true404
	br label %bb1269.outer38.backedge

cond_false439:		; preds = %cond_true404
	br i1 false, label %cond_true453, label %cond_false460

cond_true453:		; preds = %cond_false439
	br label %bb1269.outer21.backedge

cond_false460:		; preds = %cond_false439
	br i1 false, label %UnifiedReturnBlock, label %cond_false475

cond_false475:		; preds = %cond_false460
	br label %UnifiedUnreachableBlock

cond_false481:		; preds = %bb401
	br i1 false, label %cond_next496, label %bb512

cond_next496:		; preds = %cond_false481
	br i1 false, label %bb513, label %bb512

bb512:		; preds = %cond_next496, %cond_false481
	br label %bb513

bb513:		; preds = %bb512, %cond_next496
	br i1 false, label %cond_false594, label %cond_true516

cond_true516:		; preds = %bb513
	br i1 false, label %cond_true545, label %cond_false552

cond_true545:		; preds = %cond_true516
	br label %bb1269.outer38.backedge

cond_false552:		; preds = %cond_true516
	br i1 false, label %cond_true566, label %cond_false573

cond_true566:		; preds = %cond_false552
	br label %bb1269.outer21.backedge

cond_false573:		; preds = %cond_false552
	br i1 false, label %UnifiedReturnBlock, label %cond_false588

cond_false588:		; preds = %cond_false573
	br label %UnifiedUnreachableBlock

cond_false594:		; preds = %bb513
	br i1 false, label %cond_next609, label %bb625

cond_next609:		; preds = %cond_false594
	br i1 false, label %bb626, label %bb625

bb625:		; preds = %cond_next609, %cond_false594
	br label %bb626

bb626:		; preds = %bb625, %cond_next609
	br i1 false, label %cond_false707, label %cond_true629

cond_true629:		; preds = %bb626
	br i1 false, label %cond_true658, label %cond_false665

cond_true658:		; preds = %cond_true629
	br label %bb1269.outer2.backedge

bb1269.outer2.backedge:		; preds = %cond_true679, %cond_true658
	br label %bb1269.outer2

cond_false665:		; preds = %cond_true629
	br i1 false, label %cond_true679, label %cond_false686

cond_true679:		; preds = %cond_false665
	br label %bb1269.outer2.backedge

cond_false686:		; preds = %cond_false665
	br i1 false, label %UnifiedReturnBlock, label %cond_false701

cond_false701:		; preds = %cond_false686
	br label %UnifiedUnreachableBlock

cond_false707:		; preds = %bb626
	br i1 false, label %cond_next722, label %bb738

cond_next722:		; preds = %cond_false707
	br i1 false, label %bb739, label %bb738

bb738:		; preds = %cond_next722, %cond_false707
	br label %bb739

bb739:		; preds = %bb738, %cond_next722
	br i1 false, label %cond_false820, label %cond_true742

cond_true742:		; preds = %bb739
	br i1 false, label %cond_true771, label %cond_false778

cond_true771:		; preds = %cond_true742
	br label %bb1269.outer.backedge

bb1269.outer.backedge:		; preds = %cond_true792, %cond_true771
	br label %bb1269.outer

cond_false778:		; preds = %cond_true742
	br i1 false, label %cond_true792, label %cond_false799

cond_true792:		; preds = %cond_false778
	br label %bb1269.outer.backedge

cond_false799:		; preds = %cond_false778
	br i1 false, label %UnifiedReturnBlock, label %cond_false814

cond_false814:		; preds = %cond_false799
	br label %UnifiedUnreachableBlock

cond_false820:		; preds = %bb739
	br i1 false, label %cond_next835, label %bb851

cond_next835:		; preds = %cond_false820
	br i1 false, label %bb852, label %bb851

bb851:		; preds = %cond_next835, %cond_false820
	br label %bb852

bb852:		; preds = %bb851, %cond_next835
	br i1 false, label %cond_false933, label %cond_true855

cond_true855:		; preds = %bb852
	br i1 false, label %cond_true884, label %cond_false891

cond_true884:		; preds = %cond_true855
	br label %bb1269.outer.outer.backedge

bb1269.outer.outer.backedge:		; preds = %cond_true905, %cond_true884
	br label %bb1269.outer.outer

cond_false891:		; preds = %cond_true855
	br i1 false, label %cond_true905, label %cond_false912

cond_true905:		; preds = %cond_false891
	br label %bb1269.outer.outer.backedge

cond_false912:		; preds = %cond_false891
	br i1 false, label %UnifiedReturnBlock, label %cond_false927

cond_false927:		; preds = %cond_false912
	br label %UnifiedUnreachableBlock

cond_false933:		; preds = %bb852
	br i1 false, label %cond_next948, label %bb964

cond_next948:		; preds = %cond_false933
	br i1 false, label %bb965, label %bb964

bb964:		; preds = %cond_next948, %cond_false933
	br label %bb965

bb965:		; preds = %bb964, %cond_next948
	br i1 false, label %cond_false1046, label %cond_true968

cond_true968:		; preds = %bb965
	br i1 false, label %cond_true997, label %cond_false1004

cond_true997:		; preds = %cond_true968
	br label %bb1269.outer.outer.outer.backedge

bb1269.outer.outer.outer.backedge:		; preds = %cond_true1018, %cond_true997
	br label %bb1269.outer.outer.outer

cond_false1004:		; preds = %cond_true968
	br i1 false, label %cond_true1018, label %cond_false1025

cond_true1018:		; preds = %cond_false1004
	br label %bb1269.outer.outer.outer.backedge

cond_false1025:		; preds = %cond_false1004
	br i1 false, label %UnifiedReturnBlock, label %cond_false1040

cond_false1040:		; preds = %cond_false1025
	br label %UnifiedUnreachableBlock

cond_false1046:		; preds = %bb965
	br i1 false, label %cond_next1061, label %bb1077

cond_next1061:		; preds = %cond_false1046
	br i1 false, label %bb1078, label %bb1077

bb1077:		; preds = %cond_next1061, %cond_false1046
	br label %bb1078

bb1078:		; preds = %bb1077, %cond_next1061
	%tmp1080 = phi i1 [ true, %bb1077 ], [ false, %cond_next1061 ]		; <i1> [#uses=1]
	br i1 %tmp1080, label %cond_false1159, label %cond_true1081

cond_true1081:		; preds = %bb1078
	br i1 false, label %cond_true1110, label %cond_false1117

cond_true1110:		; preds = %cond_true1081
	br label %bb1269.outer.outer.outer.outer.backedge

bb1269.outer.outer.outer.outer.backedge:		; preds = %cond_true1131, %cond_true1110
	br label %bb1269.outer.outer.outer.outer

cond_false1117:		; preds = %cond_true1081
	br i1 false, label %cond_true1131, label %cond_false1138

cond_true1131:		; preds = %cond_false1117
	br label %bb1269.outer.outer.outer.outer.backedge

cond_false1138:		; preds = %cond_false1117
	br i1 false, label %UnifiedReturnBlock, label %cond_false1153

cond_false1153:		; preds = %cond_false1138
	br label %UnifiedUnreachableBlock

cond_false1159:		; preds = %bb1078
	%tmp.i119.lcssa35.lcssa.lcssa.lcssa.lcssa.lcssa = phi %"struct.kc::impl_elem_patternrepresentation"* [ null, %bb1078 ]		; <%"struct.kc::impl_elem_patternrepresentation"*> [#uses=0]
	br i1 false, label %UnifiedReturnBlock, label %cond_false1174

cond_false1174:		; preds = %cond_false1159
	br i1 false, label %UnifiedReturnBlock, label %cond_false1189

cond_false1189:		; preds = %cond_false1174
	br i1 false, label %UnifiedReturnBlock, label %cond_false1204

cond_false1204:		; preds = %cond_false1189
	br i1 false, label %UnifiedReturnBlock, label %cond_false1219

cond_false1219:		; preds = %cond_false1204
	br i1 false, label %UnifiedReturnBlock, label %cond_true1222

cond_true1222:		; preds = %cond_false1219
	br label %UnifiedReturnBlock

bb1269.outer.outer.outer.outer:		; preds = %bb1269.outer.outer.outer.outer.backedge, %entry
	br label %bb1269.outer.outer.outer

bb1269.outer.outer.outer:		; preds = %bb1269.outer.outer.outer.outer, %bb1269.outer.outer.outer.backedge
	br label %bb1269.outer.outer

bb1269.outer.outer:		; preds = %bb1269.outer.outer.outer, %bb1269.outer.outer.backedge
	br label %bb1269.outer

bb1269.outer:		; preds = %bb1269.outer.outer, %bb1269.outer.backedge
	br label %bb1269.outer2

bb1269.outer2:		; preds = %bb1269.outer, %bb1269.outer2.backedge
	br label %bb1269.outer21

bb1269.outer21:		; preds = %bb1269.outer2, %bb1269.outer21.backedge
	br label %bb1269.outer38

bb1269.outer38:		; preds = %bb1269.outer21, %bb1269.outer38.backedge
	br label %bb1269.outer54

bb1269.outer54:		; preds = %bb1269.outer38
	br label %bb1269.outer69

bb1269.outer69:		; preds = %bb1269.outer54, %cond_true
	br label %bb1269

bb1269:		; preds = %bb1269.outer69
	br i1 false, label %cond_next1281, label %bb1294

cond_next1281:		; preds = %bb1269
	br i1 false, label %cond_true1298, label %bb1294

bb1294:		; preds = %cond_next1281, %bb1269
	br i1 false, label %cond_true1331, label %cond_next1313

cond_true1298:		; preds = %cond_next1281
	br i1 false, label %cond_false, label %cond_true

cond_next1313:		; preds = %bb1294
	br i1 false, label %cond_true1331, label %cond_next1355

cond_true1331:		; preds = %cond_next1313, %bb1294
	br i1 false, label %cond_false1346, label %cond_true1342

cond_true1342:		; preds = %cond_true1331
	br label %cond_next1350

cond_false1346:		; preds = %cond_true1331
	br label %cond_next1350

cond_next1350:		; preds = %cond_false1346, %cond_true1342
	br label %bb.i

bb.i:		; preds = %bb.i, %cond_next1350
	br i1 false, label %_ZN2kc18impl_abstract_list8freelistEv.exit, label %bb.i

_ZN2kc18impl_abstract_list8freelistEv.exit:		; preds = %bb.i
	br label %cond_next1355

cond_next1355:		; preds = %_ZN2kc18impl_abstract_list8freelistEv.exit, %cond_next1313
	br i1 false, label %cond_next1363, label %bb1388

cond_next1363:		; preds = %cond_next1355
	br i1 false, label %UnifiedReturnBlock, label %cond_true1366

cond_true1366:		; preds = %cond_next1363
	br label %UnifiedReturnBlock

bb1388:		; preds = %cond_next1355
	br i1 false, label %UnifiedReturnBlock, label %bb1414.preheader

bb1414.preheader:		; preds = %bb1388
	br label %bb1414

bb1414:		; preds = %cond_true1426, %bb1414.preheader
	br i1 false, label %cond_true1426, label %bb1429

cond_true1426:		; preds = %bb1414
	br label %bb1414

bb1429:		; preds = %bb1414
	br i1 false, label %cond_true1431, label %UnifiedReturnBlock

cond_true1431:		; preds = %bb1429
	br i1 false, label %UnifiedReturnBlock, label %cond_true1434

cond_true1434:		; preds = %cond_true1431
	br label %UnifiedReturnBlock

UnifiedUnreachableBlock:		; preds = %cond_false1153, %cond_false1040, %cond_false927, %cond_false814, %cond_false701, %cond_false588, %cond_false475, %cond_false363, %cond_false250
	unreachable

UnifiedReturnBlock:		; preds = %cond_true1434, %cond_true1431, %bb1429, %bb1388, %cond_true1366, %cond_next1363, %cond_true1222, %cond_false1219, %cond_false1204, %cond_false1189, %cond_false1174, %cond_false1159, %cond_false1138, %cond_false1025, %cond_false912, %cond_false799, %cond_false686, %cond_false573, %cond_false460, %cond_false348, %cond_false235, %cond_true120, %cond_true117, %cond_true57, %cond_true55
	ret void
}
