; RUN: llvm-as < %s | opt -loop-index-split -disable-output
; PR 2030
	%struct.FULL = type { i32, i32, [1000 x float*] }

define i32 @matgen(%struct.FULL* %a, float** %x, float** %b, float** %bt, i32** %ipvt, i32 %test_case, i32 %scale) {
entry:
	br i1 false, label %bb, label %entry.bb30_crit_edge

entry.bb30_crit_edge:		; preds = %entry
	br label %bb30

bb:		; preds = %entry
	br label %bb14

bb6:		; preds = %bb14
	br label %bb14

bb14:		; preds = %bb6, %bb
	br i1 false, label %bb6, label %bb22

bb22:		; preds = %bb14
	br label %bb30

bb30:		; preds = %bb22, %entry.bb30_crit_edge
	switch i32 %test_case, label %bb648 [
		 i32 1, label %bb30.bb32_crit_edge
		 i32 2, label %bb30.bb32_crit_edge1
		 i32 3, label %bb30.bb32_crit_edge2
		 i32 4, label %bb30.bb108_crit_edge
		 i32 5, label %bb30.bb108_crit_edge3
		 i32 6, label %bb30.bb142_crit_edge
		 i32 7, label %bb30.bb142_crit_edge4
		 i32 8, label %bb30.bb142_crit_edge5
		 i32 9, label %bb234
		 i32 10, label %bb292
		 i32 11, label %bb353
		 i32 12, label %bb419
		 i32 13, label %bb485
		 i32 14, label %bb567
	]

bb30.bb142_crit_edge5:		; preds = %bb30
	br label %bb142

bb30.bb142_crit_edge4:		; preds = %bb30
	br label %bb142

bb30.bb142_crit_edge:		; preds = %bb30
	br label %bb142

bb30.bb108_crit_edge3:		; preds = %bb30
	br label %bb108

bb30.bb108_crit_edge:		; preds = %bb30
	br label %bb108

bb30.bb32_crit_edge2:		; preds = %bb30
	br label %bb32

bb30.bb32_crit_edge1:		; preds = %bb30
	br label %bb32

bb30.bb32_crit_edge:		; preds = %bb30
	br label %bb32

bb32:		; preds = %bb30.bb32_crit_edge, %bb30.bb32_crit_edge1, %bb30.bb32_crit_edge2
	br i1 false, label %bb53, label %bb52

bb52:		; preds = %bb32
	br label %bb739

bb53:		; preds = %bb32
	br label %bb101

bb58:		; preds = %bb101
	br label %bb92

bb64:		; preds = %bb92
	br i1 false, label %bb64.bb87_crit_edge, label %bb72

bb64.bb87_crit_edge:		; preds = %bb64
	br label %bb87

bb72:		; preds = %bb64
	br i1 false, label %bb72.bb87_crit_edge, label %bb79

bb72.bb87_crit_edge:		; preds = %bb72
	br label %bb87

bb79:		; preds = %bb72
	br label %bb87

bb87:		; preds = %bb79, %bb72.bb87_crit_edge, %bb64.bb87_crit_edge
	br label %bb92

bb92:		; preds = %bb87, %bb58
	br i1 false, label %bb64, label %bb98

bb98:		; preds = %bb92
	br label %bb101

bb101:		; preds = %bb98, %bb53
	br i1 false, label %bb58, label %bb107

bb107:		; preds = %bb101
	br label %bb651

bb108:		; preds = %bb30.bb108_crit_edge, %bb30.bb108_crit_edge3
	br i1 false, label %bb125, label %bb124

bb124:		; preds = %bb108
	br label %bb739

bb125:		; preds = %bb108
	br i1 false, label %bb138, label %bb139

bb138:		; preds = %bb125
	br label %bb140

bb139:		; preds = %bb125
	br label %bb140

bb140:		; preds = %bb139, %bb138
	br label %bb651

bb142:		; preds = %bb30.bb142_crit_edge, %bb30.bb142_crit_edge4, %bb30.bb142_crit_edge5
	br i1 false, label %bb161, label %bb160

bb160:		; preds = %bb142
	br label %bb739

bb161:		; preds = %bb142
	br i1 false, label %bb170, label %bb161.bb171_crit_edge

bb161.bb171_crit_edge:		; preds = %bb161
	br label %bb171

bb170:		; preds = %bb161
	br label %bb171

bb171:		; preds = %bb170, %bb161.bb171_crit_edge
	br i1 false, label %bb176, label %bb171.bb177_crit_edge

bb171.bb177_crit_edge:		; preds = %bb171
	br label %bb177

bb176:		; preds = %bb171
	br label %bb177

bb177:		; preds = %bb176, %bb171.bb177_crit_edge
	br label %bb227

bb178:		; preds = %bb227
	br label %bb218

bb184:		; preds = %bb218
	br i1 false, label %bb191, label %bb193

bb191:		; preds = %bb184
	br label %bb213

bb193:		; preds = %bb184
	br i1 false, label %bb200, label %bb203

bb200:		; preds = %bb193
	br label %bb213

bb203:		; preds = %bb193
	br i1 false, label %bb210, label %bb203.bb213_crit_edge

bb203.bb213_crit_edge:		; preds = %bb203
	br label %bb213

bb210:		; preds = %bb203
	br label %bb213

bb213:		; preds = %bb210, %bb203.bb213_crit_edge, %bb200, %bb191
	br label %bb218

bb218:		; preds = %bb213, %bb178
	br i1 false, label %bb184, label %bb224

bb224:		; preds = %bb218
	br label %bb227

bb227:		; preds = %bb224, %bb177
	br i1 false, label %bb178, label %bb233

bb233:		; preds = %bb227
	br label %bb651

bb234:		; preds = %bb30
	br i1 false, label %bb253, label %bb252

bb252:		; preds = %bb234
	br label %bb739

bb253:		; preds = %bb234
	br label %bb285

bb258:		; preds = %bb285
	br label %bb276

bb264:		; preds = %bb276
	br label %bb276

bb276:		; preds = %bb264, %bb258
	br i1 false, label %bb264, label %bb282

bb282:		; preds = %bb276
	br label %bb285

bb285:		; preds = %bb282, %bb253
	br i1 false, label %bb258, label %bb291

bb291:		; preds = %bb285
	br label %bb651

bb292:		; preds = %bb30
	br i1 false, label %bb311, label %bb310

bb310:		; preds = %bb292
	br label %bb739

bb311:		; preds = %bb292
	br label %bb346

bb316:		; preds = %bb346
	br label %bb337

bb322:		; preds = %bb337
	br label %bb337

bb337:		; preds = %bb322, %bb316
	br i1 false, label %bb322, label %bb343

bb343:		; preds = %bb337
	br label %bb346

bb346:		; preds = %bb343, %bb311
	br i1 false, label %bb316, label %bb352

bb352:		; preds = %bb346
	br label %bb651

bb353:		; preds = %bb30
	br i1 false, label %bb372, label %bb371

bb371:		; preds = %bb353
	br label %bb739

bb372:		; preds = %bb353
	br label %bb412

bb377:		; preds = %bb412
	br label %bb403

bb383:		; preds = %bb403
	br i1 false, label %bb395, label %bb389

bb389:		; preds = %bb383
	br label %bb396

bb395:		; preds = %bb383
	br label %bb396

bb396:		; preds = %bb395, %bb389
	br label %bb403

bb403:		; preds = %bb396, %bb377
	br i1 false, label %bb383, label %bb409

bb409:		; preds = %bb403
	br label %bb412

bb412:		; preds = %bb409, %bb372
	br i1 false, label %bb377, label %bb418

bb418:		; preds = %bb412
	br label %bb651

bb419:		; preds = %bb30
	br i1 false, label %bb438, label %bb437

bb437:		; preds = %bb419
	br label %bb739

bb438:		; preds = %bb419
	br label %bb478

bb443:		; preds = %bb478
	br label %bb469

bb449:		; preds = %bb469
	br i1 false, label %bb461, label %bb455

bb455:		; preds = %bb449
	br label %bb462

bb461:		; preds = %bb449
	br label %bb462

bb462:		; preds = %bb461, %bb455
	br label %bb469

bb469:		; preds = %bb462, %bb443
	br i1 false, label %bb449, label %bb475

bb475:		; preds = %bb469
	br label %bb478

bb478:		; preds = %bb475, %bb438
	br i1 false, label %bb443, label %bb484

bb484:		; preds = %bb478
	br label %bb651

bb485:		; preds = %bb30
	br i1 false, label %bb504, label %bb503

bb503:		; preds = %bb485
	br label %bb739

bb504:		; preds = %bb485
	br label %bb560

bb513:		; preds = %bb560
	br label %bb551

bb519:		; preds = %bb551
	br i1 false, label %bb528, label %bb532

bb528:		; preds = %bb519
	br label %bb536

bb532:		; preds = %bb519
	br label %bb536

bb536:		; preds = %bb532, %bb528
	br label %bb551

bb551:		; preds = %bb536, %bb513
	br i1 false, label %bb519, label %bb557

bb557:		; preds = %bb551
	br label %bb560

bb560:		; preds = %bb557, %bb504
	br i1 false, label %bb513, label %bb566

bb566:		; preds = %bb560
	br label %bb651

bb567:		; preds = %bb30
	br i1 false, label %bb586, label %bb585

bb585:		; preds = %bb567
	br label %bb739

bb586:		; preds = %bb567
	br label %bb641

bb595:		; preds = %bb641
	br label %bb632

bb601:		; preds = %bb632
	%tmp604 = icmp sgt i32 %i.7, 0		; <i1> [#uses=1]
	br i1 %tmp604, label %bb607, label %bb611

bb607:		; preds = %bb601
	br label %bb615

bb611:		; preds = %bb601
	br label %bb615

bb615:		; preds = %bb611, %bb607
	%tmp629 = add i32 %i.7, 1		; <i32> [#uses=1]
	%tmp631 = getelementptr float* %col.7, i32 1		; <float*> [#uses=1]
	br label %bb632

bb632:		; preds = %bb615, %bb595
	%col.7 = phi float* [ null, %bb595 ], [ %tmp631, %bb615 ]		; <float*> [#uses=1]
	%i.7 = phi i32 [ 0, %bb595 ], [ %tmp629, %bb615 ]		; <i32> [#uses=3]
	%tmp635 = icmp slt i32 %i.7, 0		; <i1> [#uses=1]
	br i1 %tmp635, label %bb601, label %bb638

bb638:		; preds = %bb632
	br label %bb641

bb641:		; preds = %bb638, %bb586
	br i1 false, label %bb595, label %bb647

bb647:		; preds = %bb641
	br label %bb651

bb648:		; preds = %bb30
	br label %bb739

bb651:		; preds = %bb647, %bb566, %bb484, %bb418, %bb352, %bb291, %bb233, %bb140, %bb107
	br i1 false, label %bb658, label %bb651.bb661_crit_edge

bb651.bb661_crit_edge:		; preds = %bb651
	br label %bb661

bb658:		; preds = %bb651
	br label %bb661

bb661:		; preds = %bb658, %bb651.bb661_crit_edge
	br i1 false, label %bb666, label %bb661.bb686_crit_edge

bb661.bb686_crit_edge:		; preds = %bb661
	br label %bb686

bb666:		; preds = %bb661
	br label %bb680

bb670:		; preds = %bb680
	br label %bb680

bb680:		; preds = %bb670, %bb666
	br i1 false, label %bb670, label %bb680.bb686_crit_edge

bb680.bb686_crit_edge:		; preds = %bb680
	br label %bb686

bb686:		; preds = %bb680.bb686_crit_edge, %bb661.bb686_crit_edge
	br i1 false, label %bb699, label %bb696

bb696:		; preds = %bb686
	br label %bb739

bb699:		; preds = %bb686
	br i1 false, label %bb712, label %bb709

bb709:		; preds = %bb699
	br label %bb739

bb712:		; preds = %bb699
	br i1 false, label %bb717, label %bb712.bb720_crit_edge

bb712.bb720_crit_edge:		; preds = %bb712
	br label %bb720

bb717:		; preds = %bb712
	br label %bb720

bb720:		; preds = %bb717, %bb712.bb720_crit_edge
	br i1 false, label %bb725, label %bb720.bb738_crit_edge

bb720.bb738_crit_edge:		; preds = %bb720
	br label %bb738

bb725:		; preds = %bb720
	br label %bb738

bb738:		; preds = %bb725, %bb720.bb738_crit_edge
	br label %bb739

bb739:		; preds = %bb738, %bb709, %bb696, %bb648, %bb585, %bb503, %bb437, %bb371, %bb310, %bb252, %bb160, %bb124, %bb52
	br label %return

return:		; preds = %bb739
	ret i32 0
}
