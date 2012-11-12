; RUN: opt < %s -inline -prune-eh -disable-output
; PR827
@_ZTV8CRjii = internal global [1 x i32 (...)*] [ i32 (...)* @_ZN8CRjii12NlFeeEPN5Jr7sE ]		; <[1 x i32 (...)*]*> [#uses=0]

define internal i32 @_ZN8CRjii12NlFeeEPN5Jr7sE(...) {
entry:
	br i1 false, label %cond_true, label %cond_false179

cond_true:		; preds = %entry
	br label %bb9

bb:		; preds = %cond_true14
	br label %bb9

bb9:		; preds = %bb, %cond_true
	br i1 false, label %cond_true14, label %cond_false

cond_true14:		; preds = %bb9
	br label %bb

cond_false:		; preds = %bb9
	br label %bb15

cond_next:		; No predecessors!
	br label %bb15

bb15:		; preds = %cond_next, %cond_false
	br label %bb24

bb17:		; preds = %cond_true29
	br label %bb24

bb24:		; preds = %bb17, %bb15
	br i1 false, label %cond_true29, label %cond_false30

cond_true29:		; preds = %bb24
	br label %bb17

cond_false30:		; preds = %bb24
	br label %bb32

cond_next31:		; No predecessors!
	br label %bb32

bb32:		; preds = %cond_next31, %cond_false30
	br label %bb41

bb34:		; preds = %cond_true46
	br label %bb41

bb41:		; preds = %bb34, %bb32
	br i1 false, label %cond_true46, label %cond_false47

cond_true46:		; preds = %bb41
	br label %bb34

cond_false47:		; preds = %bb41
	br label %bb49

cond_next48:		; No predecessors!
	br label %bb49

bb49:		; preds = %cond_next48, %cond_false47
	br label %bb58

bb51:		; preds = %cond_true63
	br label %bb58

bb58:		; preds = %bb51, %bb49
	br i1 false, label %cond_true63, label %cond_false64

cond_true63:		; preds = %bb58
	br label %bb51

cond_false64:		; preds = %bb58
	br label %bb66

cond_next65:		; No predecessors!
	br label %bb66

bb66:		; preds = %cond_next65, %cond_false64
	br label %bb76

bb68:		; preds = %cond_true81
	br label %bb76

bb76:		; preds = %bb68, %bb66
	br i1 false, label %cond_true81, label %cond_false82

cond_true81:		; preds = %bb76
	br label %bb68

cond_false82:		; preds = %bb76
	br label %bb84

cond_next83:		; No predecessors!
	br label %bb84

bb84:		; preds = %cond_next83, %cond_false82
	br label %bb94

bb86:		; preds = %cond_true99
	br label %bb94

bb94:		; preds = %bb86, %bb84
	br i1 false, label %cond_true99, label %cond_false100

cond_true99:		; preds = %bb94
	br label %bb86

cond_false100:		; preds = %bb94
	br label %bb102

cond_next101:		; No predecessors!
	br label %bb102

bb102:		; preds = %cond_next101, %cond_false100
	br label %bb112

bb104:		; preds = %cond_true117
	br label %bb112

bb112:		; preds = %bb104, %bb102
	br i1 false, label %cond_true117, label %cond_false118

cond_true117:		; preds = %bb112
	br label %bb104

cond_false118:		; preds = %bb112
	br label %bb120

cond_next119:		; No predecessors!
	br label %bb120

bb120:		; preds = %cond_next119, %cond_false118
	br label %bb130

bb122:		; preds = %cond_true135
	br label %bb130

bb130:		; preds = %bb122, %bb120
	br i1 false, label %cond_true135, label %cond_false136

cond_true135:		; preds = %bb130
	br label %bb122

cond_false136:		; preds = %bb130
	br label %bb138

cond_next137:		; No predecessors!
	br label %bb138

bb138:		; preds = %cond_next137, %cond_false136
	br label %bb148

bb140:		; preds = %cond_true153
	call fastcc void @_Zjrf1( )
	br label %bb148

bb148:		; preds = %bb140, %bb138
	br i1 false, label %cond_true153, label %cond_false154

cond_true153:		; preds = %bb148
	br label %bb140

cond_false154:		; preds = %bb148
	br label %bb156

cond_next155:		; No predecessors!
	br label %bb156

bb156:		; preds = %cond_next155, %cond_false154
	br label %bb166

bb158:		; preds = %cond_true171
	br label %bb166

bb166:		; preds = %bb158, %bb156
	br i1 false, label %cond_true171, label %cond_false172

cond_true171:		; preds = %bb166
	br label %bb158

cond_false172:		; preds = %bb166
	br label %bb174

cond_next173:		; No predecessors!
	br label %bb174

bb174:		; preds = %cond_next173, %cond_false172
	br label %cleanup

cleanup:		; preds = %bb174
	br label %finally

finally:		; preds = %cleanup
	br label %cond_next180

cond_false179:		; preds = %entry
	br label %cond_next180

cond_next180:		; preds = %cond_false179, %finally
	br label %return

return:		; preds = %cond_next180
	ret i32 0
}

define internal fastcc void @_Zjrf2() {
entry:
	br label %bb3

bb:		; preds = %cond_true
	br label %bb3

bb3:		; preds = %bb, %entry
	%tmp5 = load i8** null		; <i8*> [#uses=1]
	%tmp = icmp ne i8* null, %tmp5		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true, label %cond_false

cond_true:		; preds = %bb3
	br label %bb

cond_false:		; preds = %bb3
	br label %bb6

cond_next:		; No predecessors!
	br label %bb6

bb6:		; preds = %cond_next, %cond_false
	br label %return

return:		; preds = %bb6
	ret void
}

define internal fastcc void @_Zjrf3() {
entry:
	call fastcc void @_Zjrf2( )
	br label %return

return:		; preds = %entry
	ret void
}

define internal fastcc void @_Zjrf4() {
entry:
	br label %bb6

bb:		; preds = %cond_true
	br label %bb6

bb6:		; preds = %bb, %entry
	br i1 false, label %cond_true, label %cond_false

cond_true:		; preds = %bb6
	br label %bb

cond_false:		; preds = %bb6
	br label %bb8

cond_next:		; No predecessors!
	br label %bb8

bb8:		; preds = %cond_next, %cond_false
	br i1 false, label %cond_true9, label %cond_false12

cond_true9:		; preds = %bb8
	call fastcc void @_Zjrf3( )
	br label %cond_next13

cond_false12:		; preds = %bb8
	br label %cond_next13

cond_next13:		; preds = %cond_false12, %cond_true9
	br label %return

return:		; preds = %cond_next13
	ret void
}

define internal fastcc void @_Zjrf5() {
entry:
	call fastcc void @_Zjrf4( )
	br label %return

return:		; preds = %entry
	ret void
}

define internal fastcc void @_Zjrf6() {
entry:
	call fastcc void @_Zjrf5( )
	br label %return

return:		; preds = %entry
	ret void
}

define internal fastcc void @_Zjrf7() {
entry:
	br label %cleanup

cleanup:		; preds = %entry
	br label %finally

finally:		; preds = %cleanup
	call fastcc void @_Zjrf6( )
	br label %cleanup9

cleanup9:		; preds = %finally
	br label %finally8

finally8:		; preds = %cleanup9
	br label %cleanup11

cleanup11:		; preds = %finally8
	br label %finally10

finally10:		; preds = %cleanup11
	br label %finally23

finally23:		; preds = %finally10
	br label %return

return:		; preds = %finally23
	ret void
}

define internal fastcc void @_Zjrf11() {
entry:
	br label %bb7

bb:		; preds = %cond_true
	br label %bb7

bb7:		; preds = %bb, %entry
	br i1 false, label %cond_true, label %cond_false

cond_true:		; preds = %bb7
	br label %bb

cond_false:		; preds = %bb7
	br label %bb9

cond_next:		; No predecessors!
	br label %bb9

bb9:		; preds = %cond_next, %cond_false
	br label %return
		; No predecessors!
	br i1 false, label %cond_true12, label %cond_false15

cond_true12:		; preds = %0
	call fastcc void @_Zjrf3( )
	br label %cond_next16

cond_false15:		; preds = %0
	br label %cond_next16

cond_next16:		; preds = %cond_false15, %cond_true12
	br label %return

return:		; preds = %cond_next16, %bb9
	ret void
}

define internal fastcc void @_Zjrf9() {
entry:
	call fastcc void @_Zjrf11( )
	br label %return

return:		; preds = %entry
	ret void
}

define internal fastcc void @_Zjrf10() {
entry:
	call fastcc void @_Zjrf9( )
	br label %return

return:		; preds = %entry
	ret void
}

define internal fastcc void @_Zjrf8() {
entry:
	br i1 false, label %cond_true, label %cond_false201

cond_true:		; preds = %entry
	br i1 false, label %cond_true36, label %cond_false

cond_true36:		; preds = %cond_true
	br label %cleanup

cleanup:		; preds = %cond_true36
	br label %finally

finally:		; preds = %cleanup
	br label %cond_next189

cond_false:		; preds = %cond_true
	br i1 false, label %cond_true99, label %cond_false137

cond_true99:		; preds = %cond_false
	br label %cleanup136

cleanup136:		; preds = %cond_true99
	br label %finally135

finally135:		; preds = %cleanup136
	br label %cond_next

cond_false137:		; preds = %cond_false
	call fastcc void @_Zjrf10( )
	br label %cleanup188

cleanup188:		; preds = %cond_false137
	br label %finally187

finally187:		; preds = %cleanup188
	br label %cond_next

cond_next:		; preds = %finally187, %finally135
	br label %cond_next189

cond_next189:		; preds = %cond_next, %finally
	br label %cond_next202

cond_false201:		; preds = %entry
	br label %cond_next202

cond_next202:		; preds = %cond_false201, %cond_next189
	br label %return

return:		; preds = %cond_next202
	ret void
}

define internal fastcc void @_Zjrf1() {
entry:
	br label %bb492

bb:		; preds = %cond_true499
	br label %cleanup

cleanup:		; preds = %bb
	br label %finally

finally:		; preds = %cleanup
	br label %cleanup11

cleanup11:		; preds = %finally
	br label %finally10

finally10:		; preds = %cleanup11
	br i1 false, label %cond_true, label %cond_false286

cond_true:		; preds = %finally10
	br label %cleanup26

cleanup26:		; preds = %cond_true
	br label %finally25

finally25:		; preds = %cleanup26
	br label %bb30

bb27:		; preds = %cond_true37
	br label %bb30

bb30:		; preds = %bb27, %finally25
	br i1 false, label %cond_true37, label %cond_false

cond_true37:		; preds = %bb30
	br label %bb27

cond_false:		; preds = %bb30
	br label %bb38

cond_next:		; No predecessors!
	br label %bb38

bb38:		; preds = %cond_next, %cond_false
	br label %bb148

bb40:		; preds = %cond_true156
	br label %bb139

bb41:		; preds = %cond_true142
	call fastcc void @_Zjrf7( )
	br label %bb105

bb44:		; preds = %cond_true112
	br label %bb74

bb66:		; preds = %cond_true80
	br label %bb74

bb74:		; preds = %bb66, %bb44
	br i1 false, label %cond_true80, label %cond_false81

cond_true80:		; preds = %bb74
	br label %bb66

cond_false81:		; preds = %bb74
	br label %bb83

cond_next82:		; No predecessors!
	br label %bb83

bb83:		; preds = %cond_next82, %cond_false81
	br label %cleanup97

cleanup97:		; preds = %bb83
	br label %finally96

finally96:		; preds = %cleanup97
	br label %cleanup99

cleanup99:		; preds = %finally96
	br label %finally98

finally98:		; preds = %cleanup99
	br label %bb105

bb105:		; preds = %finally98, %bb41
	br i1 false, label %cond_true112, label %cond_false113

cond_true112:		; preds = %bb105
	br label %bb44

cond_false113:		; preds = %bb105
	br label %bb115

cond_next114:		; No predecessors!
	br label %bb115

bb115:		; preds = %cond_next114, %cond_false113
	br i1 false, label %cond_true119, label %cond_false123

cond_true119:		; preds = %bb115
	call fastcc void @_Zjrf8( )
	br label %cond_next124

cond_false123:		; preds = %bb115
	br label %cond_next124

cond_next124:		; preds = %cond_false123, %cond_true119
	br i1 false, label %cond_true131, label %cond_false132

cond_true131:		; preds = %cond_next124
	br label %cleanup135

cond_false132:		; preds = %cond_next124
	br label %cond_next133

cond_next133:		; preds = %cond_false132
	br label %cleanup136

cleanup135:		; preds = %cond_true131
	br label %done

cleanup136:		; preds = %cond_next133
	br label %finally134

finally134:		; preds = %cleanup136
	br label %bb139

bb139:		; preds = %finally134, %bb40
	br i1 false, label %cond_true142, label %cond_false143

cond_true142:		; preds = %bb139
	br label %bb41

cond_false143:		; preds = %bb139
	br label %bb145

cond_next144:		; No predecessors!
	br label %bb145

bb145:		; preds = %cond_next144, %cond_false143
	br label %bb148

bb148:		; preds = %bb145, %bb38
	br i1 false, label %cond_true156, label %cond_false157

cond_true156:		; preds = %bb148
	br label %bb40

cond_false157:		; preds = %bb148
	br label %bb159

cond_next158:		; No predecessors!
	br label %bb159

bb159:		; preds = %cond_next158, %cond_false157
	br label %done

done:		; preds = %bb159, %cleanup135
	br label %bb214

bb185:		; preds = %cond_true218
	br i1 false, label %cond_true193, label %cond_false206

cond_true193:		; preds = %bb185
	br label %cond_next211

cond_false206:		; preds = %bb185
	br label %cond_next211

cond_next211:		; preds = %cond_false206, %cond_true193
	br label %bb214

bb214:		; preds = %cond_next211, %done
	br i1 false, label %cond_true218, label %cond_false219

cond_true218:		; preds = %bb214
	br label %bb185

cond_false219:		; preds = %bb214
	br label %bb221

cond_next220:		; No predecessors!
	br label %bb221

bb221:		; preds = %cond_next220, %cond_false219
	br i1 false, label %cond_true236, label %cond_false245

cond_true236:		; preds = %bb221
	br label %cond_next249

cond_false245:		; preds = %bb221
	br label %cond_next249

cond_next249:		; preds = %cond_false245, %cond_true236
	br i1 false, label %cond_true272, label %cond_false277

cond_true272:		; preds = %cond_next249
	br label %cond_next278

cond_false277:		; preds = %cond_next249
	br label %cond_next278

cond_next278:		; preds = %cond_false277, %cond_true272
	br label %cleanup285

cleanup285:		; preds = %cond_next278
	br label %finally284

finally284:		; preds = %cleanup285
	br label %cond_next287

cond_false286:		; preds = %finally10
	br label %cond_next287

cond_next287:		; preds = %cond_false286, %finally284
	br i1 false, label %cond_true317, label %cond_false319

cond_true317:		; preds = %cond_next287
	br label %cond_next321

cond_false319:		; preds = %cond_next287
	br label %cond_next321

cond_next321:		; preds = %cond_false319, %cond_true317
	br label %bb348

bb335:		; preds = %cond_true355
	br label %bb348

bb348:		; preds = %bb335, %cond_next321
	br i1 false, label %cond_true355, label %cond_false356

cond_true355:		; preds = %bb348
	br label %bb335

cond_false356:		; preds = %bb348
	br label %bb358

cond_next357:		; No predecessors!
	br label %bb358

bb358:		; preds = %cond_next357, %cond_false356
	br i1 false, label %cond_true363, label %cond_false364

cond_true363:		; preds = %bb358
	br label %bb388

cond_false364:		; preds = %bb358
	br label %cond_next365

cond_next365:		; preds = %cond_false364
	br i1 false, label %cond_true370, label %cond_false371

cond_true370:		; preds = %cond_next365
	br label %bb388

cond_false371:		; preds = %cond_next365
	br label %cond_next372

cond_next372:		; preds = %cond_false371
	br i1 false, label %cond_true385, label %cond_false386

cond_true385:		; preds = %cond_next372
	br label %bb388

cond_false386:		; preds = %cond_next372
	br label %cond_next387

cond_next387:		; preds = %cond_false386
	br label %bb389

bb388:		; preds = %cond_true385, %cond_true370, %cond_true363
	br label %bb389

bb389:		; preds = %bb388, %cond_next387
	br i1 false, label %cond_true392, label %cond_false443

cond_true392:		; preds = %bb389
	br label %bb419

bb402:		; preds = %cond_true425
	br i1 false, label %cond_true406, label %cond_false412

cond_true406:		; preds = %bb402
	br label %cond_next416

cond_false412:		; preds = %bb402
	br label %cond_next416

cond_next416:		; preds = %cond_false412, %cond_true406
	br label %bb419

bb419:		; preds = %cond_next416, %cond_true392
	br i1 false, label %cond_true425, label %cond_false426

cond_true425:		; preds = %bb419
	br label %bb402

cond_false426:		; preds = %bb419
	br label %bb428

cond_next427:		; No predecessors!
	br label %bb428

bb428:		; preds = %cond_next427, %cond_false426
	br label %cond_next478

cond_false443:		; preds = %bb389
	br label %bb460

bb450:		; preds = %cond_true466
	br label %bb460

bb460:		; preds = %bb450, %cond_false443
	br i1 false, label %cond_true466, label %cond_false467

cond_true466:		; preds = %bb460
	br label %bb450

cond_false467:		; preds = %bb460
	br label %bb469

cond_next468:		; No predecessors!
	br label %bb469

bb469:		; preds = %cond_next468, %cond_false467
	br label %cond_next478

cond_next478:		; preds = %bb469, %bb428
	br label %cleanup485

cleanup485:		; preds = %cond_next478
	br label %finally484

finally484:		; preds = %cleanup485
	br label %cleanup487

cleanup487:		; preds = %finally484
	br label %finally486

finally486:		; preds = %cleanup487
	br label %cleanup489

cleanup489:		; preds = %finally486
	br label %finally488

finally488:		; preds = %cleanup489
	br label %bb492

bb492:		; preds = %finally488, %entry
	br i1 false, label %cond_true499, label %cond_false500

cond_true499:		; preds = %bb492
	br label %bb

cond_false500:		; preds = %bb492
	br label %bb502

cond_next501:		; No predecessors!
	br label %bb502

bb502:		; preds = %cond_next501, %cond_false500
	br label %return

return:		; preds = %bb502
	ret void
}

define internal fastcc void @_ZSt26__unguarded_insertion_sortIN9__gnu_cxx17__normal_iteratorIPSsSt6vectorISsSaISsEEEEEvT_S7_() {
entry:
	br label %bb12

bb:		; preds = %cond_true
	br label %cleanup

cleanup:		; preds = %bb
	br label %finally

finally:		; preds = %cleanup
	br label %bb12

bb12:		; preds = %finally, %entry
	br i1 false, label %cond_true, label %cond_false

cond_true:		; preds = %bb12
	br label %bb

cond_false:		; preds = %bb12
	br label %bb14

cond_next:		; No predecessors!
	br label %bb14

bb14:		; preds = %cond_next, %cond_false
	br label %return

return:		; preds = %bb14
	ret void
}
