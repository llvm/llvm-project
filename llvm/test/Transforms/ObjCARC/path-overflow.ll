; RUN: opt -passes=objc-arc -S < %s
; rdar://12277446
; rdar://12480535
; rdar://14590914
; rdar://15377890

; The total number of paths grows exponentially with the number of branches, and a
; computation of this number can overflow any reasonable fixed-sized
; integer. This can occur in both the addition phase when we are adding up the
; total bottomup/topdown paths and when we multiply them together at the end.

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.0.0"

%struct.NSConstantString = type { i32*, i32, i8*, i32 }
%struct.CGPoint = type { float, float }

@_unnamed_cfstring = external constant %struct.NSConstantString, section "__DATA,__cfstring"
@_unnamed_cfstring_2 = external constant %struct.NSConstantString, section "__DATA,__cfstring"

declare i8* @llvm.objc.retain(i8*) nonlazybind
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*) nonlazybind
declare void @llvm.objc.release(i8*) nonlazybind
declare i8* @returner()
declare i8* @objc_msgSend(i8*, i8*, ...) nonlazybind
declare void @NSLog(i8*, ...)
declare void @objc_msgSend_stret(i8*, i8*, ...)
declare i32 @__gxx_personality_sj0(...)
declare i32 @__objc_personality_v0(...)


define hidden void @test1() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  br i1 undef, label %msgSend.nullinit, label %msgSend.call

msgSend.call:                                     ; preds = %entry
  br label %msgSend.cont

msgSend.nullinit:                                 ; preds = %entry
  br label %msgSend.cont

msgSend.cont:                                     ; preds = %msgSend.nullinit, %msgSend.call
  %0 = bitcast %struct.NSConstantString* @_unnamed_cfstring to i8*
  %1 = call i8* @llvm.objc.retain(i8* %0) nounwind
  br i1 undef, label %msgSend.nullinit33, label %msgSend.call32

msgSend.call32:                                   ; preds = %if.end10
  br label %msgSend.cont34

msgSend.nullinit33:                               ; preds = %if.end10
  br label %msgSend.cont34

msgSend.cont34:                                   ; preds = %msgSend.nullinit33, %msgSend.call32
  br i1 undef, label %msgSend.nullinit38, label %msgSend.call37

msgSend.call37:                                   ; preds = %msgSend.cont34
  br label %msgSend.cont39

msgSend.nullinit38:                               ; preds = %msgSend.cont34
  br label %msgSend.cont39

msgSend.cont39:                                   ; preds = %msgSend.nullinit38, %msgSend.call37
  br i1 undef, label %msgSend.nullinit49, label %msgSend.call48

msgSend.call48:                                   ; preds = %msgSend.cont39
  br label %msgSend.cont50

msgSend.nullinit49:                               ; preds = %msgSend.cont39
  br label %msgSend.cont50

msgSend.cont50:                                   ; preds = %msgSend.nullinit49, %msgSend.call48
  br i1 undef, label %msgSend.nullinit61, label %msgSend.call60

msgSend.call60:                                   ; preds = %msgSend.cont50
  br label %msgSend.cont62

msgSend.nullinit61:                               ; preds = %msgSend.cont50
  br label %msgSend.cont62

msgSend.cont62:                                   ; preds = %msgSend.nullinit61, %msgSend.call60
  br i1 undef, label %msgSend.nullinit67, label %msgSend.call66

msgSend.call66:                                   ; preds = %msgSend.cont62
  br label %msgSend.cont68

msgSend.nullinit67:                               ; preds = %msgSend.cont62
  br label %msgSend.cont68

msgSend.cont68:                                   ; preds = %msgSend.nullinit67, %msgSend.call66
  br i1 undef, label %msgSend.nullinit84, label %msgSend.call83

msgSend.call83:                                   ; preds = %msgSend.cont68
  br label %msgSend.cont85

msgSend.nullinit84:                               ; preds = %msgSend.cont68
  br label %msgSend.cont85

msgSend.cont85:                                   ; preds = %msgSend.nullinit84, %msgSend.call83
  br i1 undef, label %msgSend.nullinit90, label %msgSend.call89

msgSend.call89:                                   ; preds = %msgSend.cont85
  br label %msgSend.cont91

msgSend.nullinit90:                               ; preds = %msgSend.cont85
  br label %msgSend.cont91

msgSend.cont91:                                   ; preds = %msgSend.nullinit90, %msgSend.call89
  br i1 undef, label %msgSend.nullinit104, label %msgSend.call103

msgSend.call103:                                  ; preds = %msgSend.cont91
  br label %msgSend.cont105

msgSend.nullinit104:                              ; preds = %msgSend.cont91
  br label %msgSend.cont105

msgSend.cont105:                                  ; preds = %msgSend.nullinit104, %msgSend.call103
  br i1 undef, label %land.lhs.true, label %if.end123

land.lhs.true:                                    ; preds = %msgSend.cont105
  br i1 undef, label %if.then117, label %if.end123

if.then117:                                       ; preds = %land.lhs.true
  br label %if.end123

if.end123:                                        ; preds = %if.then117, %land.lhs.true, %msgSend.cont105
  br i1 undef, label %msgSend.nullinit132, label %msgSend.call131

msgSend.call131:                                  ; preds = %if.end123
  br label %msgSend.cont133

msgSend.nullinit132:                              ; preds = %if.end123
  br label %msgSend.cont133

msgSend.cont133:                                  ; preds = %msgSend.nullinit132, %msgSend.call131
  br i1 undef, label %msgSend.nullinit139, label %msgSend.call138

msgSend.call138:                                  ; preds = %msgSend.cont133
  br label %msgSend.cont140

msgSend.nullinit139:                              ; preds = %msgSend.cont133
  br label %msgSend.cont140

msgSend.cont140:                                  ; preds = %msgSend.nullinit139, %msgSend.call138
  br i1 undef, label %if.then151, label %if.end157

if.then151:                                       ; preds = %msgSend.cont140
  br label %if.end157

if.end157:                                        ; preds = %if.then151, %msgSend.cont140
  br i1 undef, label %msgSend.nullinit164, label %msgSend.call163

msgSend.call163:                                  ; preds = %if.end157
  br label %msgSend.cont165

msgSend.nullinit164:                              ; preds = %if.end157
  br label %msgSend.cont165

msgSend.cont165:                                  ; preds = %msgSend.nullinit164, %msgSend.call163
  br i1 undef, label %msgSend.nullinit176, label %msgSend.call175

msgSend.call175:                                  ; preds = %msgSend.cont165
  br label %msgSend.cont177

msgSend.nullinit176:                              ; preds = %msgSend.cont165
  br label %msgSend.cont177

msgSend.cont177:                                  ; preds = %msgSend.nullinit176, %msgSend.call175
  br i1 undef, label %land.lhs.true181, label %if.end202

land.lhs.true181:                                 ; preds = %msgSend.cont177
  br i1 undef, label %if.then187, label %if.end202

if.then187:                                       ; preds = %land.lhs.true181
  br i1 undef, label %msgSend.nullinit199, label %msgSend.call198

msgSend.call198:                                  ; preds = %if.then187
  br label %msgSend.cont200

msgSend.nullinit199:                              ; preds = %if.then187
  br label %msgSend.cont200

msgSend.cont200:                                  ; preds = %msgSend.nullinit199, %msgSend.call198
  br label %if.end202

if.end202:                                        ; preds = %msgSend.cont200, %land.lhs.true181, %msgSend.cont177
  br i1 undef, label %msgSend.nullinit236, label %msgSend.call235

msgSend.call235:                                  ; preds = %if.end202
  br label %msgSend.cont237

msgSend.nullinit236:                              ; preds = %if.end202
  br label %msgSend.cont237

msgSend.cont237:                                  ; preds = %msgSend.nullinit236, %msgSend.call235
  br i1 undef, label %msgSend.nullinit254, label %msgSend.call253

msgSend.call253:                                  ; preds = %msgSend.cont237
  br label %msgSend.cont255

msgSend.nullinit254:                              ; preds = %msgSend.cont237
  br label %msgSend.cont255

msgSend.cont255:                                  ; preds = %msgSend.nullinit254, %msgSend.call253
  br i1 undef, label %msgSend.nullinit269, label %msgSend.call268

msgSend.call268:                                  ; preds = %msgSend.cont255
  br label %msgSend.cont270

msgSend.nullinit269:                              ; preds = %msgSend.cont255
  br label %msgSend.cont270

msgSend.cont270:                                  ; preds = %msgSend.nullinit269, %msgSend.call268
  br i1 undef, label %msgSend.nullinit281, label %msgSend.call280

msgSend.call280:                                  ; preds = %msgSend.cont270
  br label %msgSend.cont282

msgSend.nullinit281:                              ; preds = %msgSend.cont270
  br label %msgSend.cont282

msgSend.cont282:                                  ; preds = %msgSend.nullinit281, %msgSend.call280
  br i1 undef, label %msgSend.nullinit287, label %msgSend.call286

msgSend.call286:                                  ; preds = %msgSend.cont282
  br label %msgSend.cont288

msgSend.nullinit287:                              ; preds = %msgSend.cont282
  br label %msgSend.cont288

msgSend.cont288:                                  ; preds = %msgSend.nullinit287, %msgSend.call286
  br i1 undef, label %msgSend.nullinit303, label %msgSend.call302

msgSend.call302:                                  ; preds = %msgSend.cont288
  br label %msgSend.cont304

msgSend.nullinit303:                              ; preds = %msgSend.cont288
  br label %msgSend.cont304

msgSend.cont304:                                  ; preds = %msgSend.nullinit303, %msgSend.call302
  br i1 undef, label %msgSend.nullinit344, label %msgSend.call343

msgSend.call343:                                  ; preds = %msgSend.cont304
  br label %msgSend.cont345

msgSend.nullinit344:                              ; preds = %msgSend.cont304
  br label %msgSend.cont345

msgSend.cont345:                                  ; preds = %msgSend.nullinit344, %msgSend.call343
  br i1 undef, label %msgSend.nullinit350, label %msgSend.call349

msgSend.call349:                                  ; preds = %msgSend.cont345
  br label %msgSend.cont351

msgSend.nullinit350:                              ; preds = %msgSend.cont345
  br label %msgSend.cont351

msgSend.cont351:                                  ; preds = %msgSend.nullinit350, %msgSend.call349
  br i1 undef, label %msgSend.nullinit366, label %msgSend.call365

msgSend.call365:                                  ; preds = %msgSend.cont351
  br label %msgSend.cont367

msgSend.nullinit366:                              ; preds = %msgSend.cont351
  br label %msgSend.cont367

msgSend.cont367:                                  ; preds = %msgSend.nullinit366, %msgSend.call365
  br i1 undef, label %msgSend.nullinit376, label %msgSend.call375

msgSend.call375:                                  ; preds = %msgSend.cont367
  br label %msgSend.cont377

msgSend.nullinit376:                              ; preds = %msgSend.cont367
  br label %msgSend.cont377

msgSend.cont377:                                  ; preds = %msgSend.nullinit376, %msgSend.call375
  br i1 undef, label %if.then384, label %if.else401

if.then384:                                       ; preds = %msgSend.cont377
  br i1 undef, label %msgSend.nullinit392, label %msgSend.call391

msgSend.call391:                                  ; preds = %if.then384
  br label %msgSend.cont393

msgSend.nullinit392:                              ; preds = %if.then384
  br label %msgSend.cont393

msgSend.cont393:                                  ; preds = %msgSend.nullinit392, %msgSend.call391
  br label %if.end418

if.else401:                                       ; preds = %msgSend.cont377
  br i1 undef, label %msgSend.nullinit409, label %msgSend.call408

msgSend.call408:                                  ; preds = %if.else401
  br label %msgSend.cont410

msgSend.nullinit409:                              ; preds = %if.else401
  br label %msgSend.cont410

msgSend.cont410:                                  ; preds = %msgSend.nullinit409, %msgSend.call408
  br label %if.end418

if.end418:                                        ; preds = %msgSend.cont410, %msgSend.cont393
  br i1 undef, label %msgSend.nullinit470, label %msgSend.call469

msgSend.call469:                                  ; preds = %if.end418
  br label %msgSend.cont471

msgSend.nullinit470:                              ; preds = %if.end418
  br label %msgSend.cont471

msgSend.cont471:                                  ; preds = %msgSend.nullinit470, %msgSend.call469
  br i1 undef, label %msgSend.nullinit484, label %msgSend.call483

msgSend.call483:                                  ; preds = %msgSend.cont471
  br label %msgSend.cont485

msgSend.nullinit484:                              ; preds = %msgSend.cont471
  br label %msgSend.cont485

msgSend.cont485:                                  ; preds = %msgSend.nullinit484, %msgSend.call483
  br i1 undef, label %msgSend.nullinit500, label %msgSend.call499

msgSend.call499:                                  ; preds = %msgSend.cont485
  br label %msgSend.cont501

msgSend.nullinit500:                              ; preds = %msgSend.cont485
  br label %msgSend.cont501

msgSend.cont501:                                  ; preds = %msgSend.nullinit500, %msgSend.call499
  br i1 undef, label %msgSend.nullinit506, label %msgSend.call505

msgSend.call505:                                  ; preds = %msgSend.cont501
  br label %msgSend.cont507

msgSend.nullinit506:                              ; preds = %msgSend.cont501
  br label %msgSend.cont507

msgSend.cont507:                                  ; preds = %msgSend.nullinit506, %msgSend.call505
  call void @llvm.objc.release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Function Attrs: optsize ssp uwtable
define void @test2() unnamed_addr align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
bb:
  br i1 undef, label %bb3, label %bb2

bb2:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  br i1 undef, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  br label %bb5

bb5:                                              ; preds = %bb4, %bb3
  br i1 undef, label %bb7, label %bb6

bb6:                                              ; preds = %bb5
  br label %bb7

bb7:                                              ; preds = %bb6, %bb5
  br i1 undef, label %bb9, label %bb8

bb8:                                              ; preds = %bb7
  unreachable

bb9:                                              ; preds = %bb7
  br i1 undef, label %bb11, label %bb10

bb10:                                             ; preds = %bb9
  br label %bb11

bb11:                                             ; preds = %bb10, %bb9
  br i1 undef, label %bb13, label %bb12

bb12:                                             ; preds = %bb11
  br label %bb13

bb13:                                             ; preds = %bb12, %bb11
  br i1 undef, label %bb15, label %bb14

bb14:                                             ; preds = %bb13
  br label %bb15

bb15:                                             ; preds = %bb14, %bb13
  br i1 undef, label %bb17, label %bb16

bb16:                                             ; preds = %bb15
  br label %bb17

bb17:                                             ; preds = %bb16, %bb15
  br i1 undef, label %bb19, label %bb18

bb18:                                             ; preds = %bb17
  br label %bb19

bb19:                                             ; preds = %bb18, %bb17
  br i1 undef, label %bb222, label %bb20

bb20:                                             ; preds = %bb19
  br i1 undef, label %bb222, label %bb21

bb21:                                             ; preds = %bb20
  br i1 undef, label %bb22, label %bb30

bb22:                                             ; preds = %bb21
  br i1 undef, label %bb23, label %bb32

bb23:                                             ; preds = %bb22
  br i1 undef, label %bb24, label %bb34

bb24:                                             ; preds = %bb23
  br i1 undef, label %bb26, label %bb25

bb25:                                             ; preds = %bb24
  br label %bb27

bb26:                                             ; preds = %bb24
  br label %bb27

bb27:                                             ; preds = %bb26, %bb25
  br i1 undef, label %bb28, label %bb42

bb28:                                             ; preds = %bb27
  br i1 undef, label %bb36, label %bb29

bb29:                                             ; preds = %bb28
  br label %bb36

bb30:                                             ; preds = %bb210, %bb207, %bb203, %bb199, %bb182, %bb176, %bb174, %bb171, %bb136, %bb132, %bb21
  br label %bb213

bb32:                                             ; preds = %bb22
  unreachable

bb34:                                             ; preds = %bb23
  unreachable

bb36:                                             ; preds = %bb29, %bb28
  br i1 undef, label %bb38, label %bb37

bb37:                                             ; preds = %bb36
  br label %bb39

bb38:                                             ; preds = %bb36
  br label %bb39

bb39:                                             ; preds = %bb38, %bb37
  br i1 undef, label %bb41, label %bb40

bb40:                                             ; preds = %bb39
  unreachable

bb41:                                             ; preds = %bb39
  br label %bb42

bb42:                                             ; preds = %bb41, %bb27
  br i1 undef, label %bb43, label %bb214

bb43:                                             ; preds = %bb42
  br i1 undef, label %bb47, label %bb45

bb45:                                             ; preds = %bb130, %bb128, %bb126, %bb124, %bb122, %bb120, %bb118, %bb116, %bb114, %bb112, %bb110, %bb108, %bb105, %bb102, %bb100, %bb96, %bb94, %bb90, %bb88, %bb84, %bb82, %bb78, %bb76, %bb72, %bb70, %bb66, %bb64, %bb60, %bb58, %bb54, %bb51, %bb43
  unreachable

bb47:                                             ; preds = %bb43
  br i1 undef, label %bb48, label %bb106

bb48:                                             ; preds = %bb47
  br i1 undef, label %bb50, label %bb49

bb49:                                             ; preds = %bb48
  br label %bb51

bb50:                                             ; preds = %bb48
  br label %bb51

bb51:                                             ; preds = %bb50, %bb49
  br i1 undef, label %bb53, label %bb45

bb53:                                             ; preds = %bb51
  br i1 undef, label %bb54, label %bb134

bb54:                                             ; preds = %bb53
  br i1 undef, label %bb55, label %bb45

bb55:                                             ; preds = %bb54
  br i1 undef, label %bb57, label %bb56

bb56:                                             ; preds = %bb55
  br label %bb58

bb57:                                             ; preds = %bb55
  br label %bb58

bb58:                                             ; preds = %bb57, %bb56
  br i1 undef, label %bb60, label %bb45

bb60:                                             ; preds = %bb58
  br i1 undef, label %bb61, label %bb45

bb61:                                             ; preds = %bb60
  br i1 undef, label %bb63, label %bb62

bb62:                                             ; preds = %bb61
  br label %bb64

bb63:                                             ; preds = %bb61
  br label %bb64

bb64:                                             ; preds = %bb63, %bb62
  br i1 undef, label %bb66, label %bb45

bb66:                                             ; preds = %bb64
  br i1 undef, label %bb67, label %bb45

bb67:                                             ; preds = %bb66
  br i1 undef, label %bb69, label %bb68

bb68:                                             ; preds = %bb67
  br label %bb70

bb69:                                             ; preds = %bb67
  br label %bb70

bb70:                                             ; preds = %bb69, %bb68
  br i1 undef, label %bb72, label %bb45

bb72:                                             ; preds = %bb70
  br i1 undef, label %bb73, label %bb45

bb73:                                             ; preds = %bb72
  br i1 undef, label %bb75, label %bb74

bb74:                                             ; preds = %bb73
  br label %bb76

bb75:                                             ; preds = %bb73
  br label %bb76

bb76:                                             ; preds = %bb75, %bb74
  br i1 undef, label %bb78, label %bb45

bb78:                                             ; preds = %bb76
  br i1 undef, label %bb79, label %bb45

bb79:                                             ; preds = %bb78
  br i1 undef, label %bb81, label %bb80

bb80:                                             ; preds = %bb79
  br label %bb82

bb81:                                             ; preds = %bb79
  br label %bb82

bb82:                                             ; preds = %bb81, %bb80
  br i1 undef, label %bb84, label %bb45

bb84:                                             ; preds = %bb82
  br i1 undef, label %bb85, label %bb45

bb85:                                             ; preds = %bb84
  br i1 undef, label %bb87, label %bb86

bb86:                                             ; preds = %bb85
  br label %bb88

bb87:                                             ; preds = %bb85
  br label %bb88

bb88:                                             ; preds = %bb87, %bb86
  br i1 undef, label %bb90, label %bb45

bb90:                                             ; preds = %bb88
  br i1 undef, label %bb91, label %bb45

bb91:                                             ; preds = %bb90
  br i1 undef, label %bb93, label %bb92

bb92:                                             ; preds = %bb91
  br label %bb94

bb93:                                             ; preds = %bb91
  br label %bb94

bb94:                                             ; preds = %bb93, %bb92
  br i1 undef, label %bb96, label %bb45

bb96:                                             ; preds = %bb94
  br i1 undef, label %bb97, label %bb45

bb97:                                             ; preds = %bb96
  br i1 undef, label %bb99, label %bb98

bb98:                                             ; preds = %bb97
  br label %bb100

bb99:                                             ; preds = %bb97
  br label %bb100

bb100:                                            ; preds = %bb99, %bb98
  br i1 undef, label %bb102, label %bb45

bb102:                                            ; preds = %bb100
  br i1 undef, label %bb104, label %bb45

bb104:                                            ; preds = %bb102
  br i1 undef, label %bb108, label %bb105

bb105:                                            ; preds = %bb104
  br i1 undef, label %bb108, label %bb45

bb106:                                            ; preds = %bb47
  unreachable

bb108:                                            ; preds = %bb105, %bb104
  br i1 undef, label %bb110, label %bb45

bb110:                                            ; preds = %bb108
  br i1 undef, label %bb112, label %bb45

bb112:                                            ; preds = %bb110
  br i1 undef, label %bb114, label %bb45

bb114:                                            ; preds = %bb112
  br i1 undef, label %bb116, label %bb45

bb116:                                            ; preds = %bb114
  br i1 undef, label %bb118, label %bb45

bb118:                                            ; preds = %bb116
  br i1 undef, label %bb120, label %bb45

bb120:                                            ; preds = %bb118
  br i1 undef, label %bb122, label %bb45

bb122:                                            ; preds = %bb120
  br i1 undef, label %bb124, label %bb45

bb124:                                            ; preds = %bb122
  br i1 undef, label %bb126, label %bb45

bb126:                                            ; preds = %bb124
  br i1 undef, label %bb128, label %bb45

bb128:                                            ; preds = %bb126
  br i1 undef, label %bb130, label %bb45

bb130:                                            ; preds = %bb128
  br i1 undef, label %bb132, label %bb45

bb132:                                            ; preds = %bb130
  br i1 undef, label %bb135, label %bb30

bb134:                                            ; preds = %bb53
  unreachable

bb135:                                            ; preds = %bb132
  br i1 undef, label %bb139, label %bb136

bb136:                                            ; preds = %bb135
  br i1 undef, label %bb138, label %bb30

bb138:                                            ; preds = %bb136
  br label %bb139

bb139:                                            ; preds = %bb138, %bb135
  br i1 undef, label %bb140, label %bb141

bb140:                                            ; preds = %bb139
  unreachable

bb141:                                            ; preds = %bb139
  br i1 undef, label %bb142, label %bb215

bb142:                                            ; preds = %bb141
  br i1 undef, label %bb144, label %bb143

bb143:                                            ; preds = %bb142
  br label %bb145

bb144:                                            ; preds = %bb142
  br label %bb145

bb145:                                            ; preds = %bb144, %bb143
  br i1 undef, label %bb146, label %bb151

bb146:                                            ; preds = %bb145
  br i1 undef, label %bb148, label %bb153

bb148:                                            ; preds = %bb146
  br i1 undef, label %bb155, label %bb149

bb149:                                            ; preds = %bb148
  br i1 undef, label %bb150, label %bb153

bb150:                                            ; preds = %bb149
  br label %bb155

bb151:                                            ; preds = %bb145
  unreachable

bb153:                                            ; preds = %bb158, %bb149, %bb146
  unreachable

bb155:                                            ; preds = %bb150, %bb148
  br i1 undef, label %bb157, label %bb156

bb156:                                            ; preds = %bb155
  br label %bb158

bb157:                                            ; preds = %bb155
  br label %bb158

bb158:                                            ; preds = %bb157, %bb156
  br i1 undef, label %bb160, label %bb153

bb160:                                            ; preds = %bb158
  br i1 undef, label %bb162, label %bb161

bb161:                                            ; preds = %bb160
  br label %bb163

bb162:                                            ; preds = %bb160
  br label %bb163

bb163:                                            ; preds = %bb162, %bb161
  br i1 undef, label %bb165, label %bb164

bb164:                                            ; preds = %bb163
  br label %bb165

bb165:                                            ; preds = %bb164, %bb163
  br i1 undef, label %bb170, label %bb166

bb166:                                            ; preds = %bb165
  br i1 undef, label %bb167, label %bb168

bb167:                                            ; preds = %bb166
  unreachable

bb168:                                            ; preds = %bb166
  unreachable

bb170:                                            ; preds = %bb165
  br i1 undef, label %bb215, label %bb171

bb171:                                            ; preds = %bb170
  br i1 undef, label %bb173, label %bb30

bb173:                                            ; preds = %bb171
  br i1 undef, label %bb174, label %bb215

bb174:                                            ; preds = %bb173
  br i1 undef, label %bb176, label %bb30

bb176:                                            ; preds = %bb174
  br i1 undef, label %bb178, label %bb30

bb178:                                            ; preds = %bb176
  br i1 undef, label %bb179, label %bb193

bb179:                                            ; preds = %bb178
  br i1 undef, label %bb181, label %bb180

bb180:                                            ; preds = %bb179
  br label %bb182

bb181:                                            ; preds = %bb179
  br label %bb182

bb182:                                            ; preds = %bb181, %bb180
  br i1 undef, label %bb184, label %bb30

bb184:                                            ; preds = %bb182
  %tmp185 = call i8* @returner()
  br i1 undef, label %bb186, label %bb195

bb186:                                            ; preds = %bb184
  %tmp188 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %tmp185)
  %tmp189 = call i8* @llvm.objc.retain(i8* %tmp188)
  call void @llvm.objc.release(i8* %tmp189), !clang.imprecise_release !0
  br i1 undef, label %bb197, label %bb190

bb190:                                            ; preds = %bb186
  br i1 undef, label %bb192, label %bb195

bb192:                                            ; preds = %bb190
  br i1 undef, label %bb197, label %bb195

bb193:                                            ; preds = %bb178
  br label %bb213

bb195:                                            ; preds = %bb192, %bb190, %bb184
  unreachable

bb197:                                            ; preds = %bb192, %bb186
  br i1 undef, label %bb198, label %bb215

bb198:                                            ; preds = %bb197
  br i1 undef, label %bb202, label %bb199

bb199:                                            ; preds = %bb198
  br i1 undef, label %bb201, label %bb30

bb201:                                            ; preds = %bb199
  br label %bb202

bb202:                                            ; preds = %bb201, %bb198
  br i1 undef, label %bb206, label %bb203

bb203:                                            ; preds = %bb202
  br i1 undef, label %bb205, label %bb30

bb205:                                            ; preds = %bb203
  br label %bb206

bb206:                                            ; preds = %bb205, %bb202
  br i1 undef, label %bb210, label %bb207

bb207:                                            ; preds = %bb206
  br i1 undef, label %bb209, label %bb30

bb209:                                            ; preds = %bb207
  br label %bb210

bb210:                                            ; preds = %bb209, %bb206
  br i1 undef, label %bb212, label %bb30

bb212:                                            ; preds = %bb210
  unreachable

bb213:                                            ; preds = %bb193, %bb30
  resume { i8*, i32 } undef

bb214:                                            ; preds = %bb42
  br label %bb219

bb215:                                            ; preds = %bb197, %bb173, %bb170, %bb141
  br i1 undef, label %bb217, label %bb216

bb216:                                            ; preds = %bb215
  br label %bb217

bb217:                                            ; preds = %bb216, %bb215
  br i1 undef, label %bb219, label %bb218

bb218:                                            ; preds = %bb217
  br label %bb219

bb219:                                            ; preds = %bb218, %bb217, %bb214
  br i1 undef, label %bb221, label %bb220

bb220:                                            ; preds = %bb219
  unreachable

bb221:                                            ; preds = %bb219
  unreachable

bb222:                                            ; preds = %bb20, %bb19
  ret void
}

; Function Attrs: ssp
define void @test3() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  %call2 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %call5 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont4 unwind label %lpad3

invoke.cont4:                                     ; preds = %invoke.cont
  br i1 undef, label %land.end, label %land.rhs

land.rhs:                                         ; preds = %invoke.cont4
  %call7 = invoke i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*)*)(i8* undef, i8* undef)
          to label %land.end unwind label %lpad3

land.end:                                         ; preds = %land.rhs, %invoke.cont4
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i unwind label %lpad.i

invoke.cont.i:                                    ; preds = %land.end
  br i1 undef, label %invoke.cont8, label %if.then.i

if.then.i:                                        ; preds = %invoke.cont.i
  br label %invoke.cont8

lpad.i:                                           ; preds = %land.end
  %tmp13 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont8:                                     ; preds = %if.then.i, %invoke.cont.i
  %call18 = invoke i8* (i8*, i8*, i8*, ...) bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*, ...)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef)
          to label %invoke.cont17 unwind label %lpad16

invoke.cont17:                                    ; preds = %invoke.cont8
  %call22 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont21 unwind label %lpad20

invoke.cont21:                                    ; preds = %invoke.cont17
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i1980 unwind label %lpad.i1982

invoke.cont.i1980:                                ; preds = %invoke.cont21
  br i1 undef, label %invoke.cont24, label %if.then.i1981

if.then.i1981:                                    ; preds = %invoke.cont.i1980
  br label %invoke.cont24

lpad.i1982:                                       ; preds = %invoke.cont21
  %tmp28 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont24:                                    ; preds = %if.then.i1981, %invoke.cont.i1980
  %call37 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont36 unwind label %lpad35

invoke.cont36:                                    ; preds = %invoke.cont24
  br i1 undef, label %land.end43, label %land.rhs39

land.rhs39:                                       ; preds = %invoke.cont36
  %call41 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %land.end43 unwind label %lpad35

land.end43:                                       ; preds = %land.rhs39, %invoke.cont36
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i1986 unwind label %lpad.i1988

invoke.cont.i1986:                                ; preds = %land.end43
  br i1 undef, label %invoke.cont44, label %if.then.i1987

if.then.i1987:                                    ; preds = %invoke.cont.i1986
  br label %invoke.cont44

lpad.i1988:                                       ; preds = %land.end43
  %tmp42 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont44:                                    ; preds = %if.then.i1987, %invoke.cont.i1986
  %call53 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont52 unwind label %lpad51

invoke.cont52:                                    ; preds = %invoke.cont44
  br i1 undef, label %land.end70, label %land.rhs58

land.rhs58:                                       ; preds = %invoke.cont52
  %call63 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 42)
          to label %invoke.cont62 unwind label %lpad61

invoke.cont62:                                    ; preds = %land.rhs58
  %call68 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef)
          to label %land.end70 unwind label %lpad66.body.thread

land.end70:                                       ; preds = %invoke.cont62, %invoke.cont52
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i1992 unwind label %lpad66.body

invoke.cont.i1992:                                ; preds = %land.end70
  br i1 undef, label %invoke.cont71, label %if.then.i1993

if.then.i1993:                                    ; preds = %invoke.cont.i1992
  br label %invoke.cont71

invoke.cont71:                                    ; preds = %if.then.i1993, %invoke.cont.i1992
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i1998 unwind label %lpad.i2000

invoke.cont.i1998:                                ; preds = %invoke.cont71
  br i1 undef, label %invoke.cont91, label %if.then.i1999

if.then.i1999:                                    ; preds = %invoke.cont.i1998
  br label %invoke.cont91

lpad.i2000:                                       ; preds = %invoke.cont71
  %tmp74 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup102

invoke.cont91:                                    ; preds = %if.then.i1999, %invoke.cont.i1998
  %call96 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont95 unwind label %lpad94

invoke.cont95:                                    ; preds = %invoke.cont91
  %call98 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* %call96)
          to label %invoke.cont97 unwind label %lpad94

invoke.cont97:                                    ; preds = %invoke.cont95
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2004 unwind label %lpad.i2006

invoke.cont.i2004:                                ; preds = %invoke.cont97
  br i1 undef, label %invoke.cont100, label %if.then.i2005

if.then.i2005:                                    ; preds = %invoke.cont.i2004
  br label %invoke.cont100

lpad.i2006:                                       ; preds = %invoke.cont97
  %tmp82 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont100:                                   ; preds = %if.then.i2005, %invoke.cont.i2004
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont110 unwind label %lpad109

invoke.cont110:                                   ; preds = %invoke.cont100
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2010 unwind label %lpad.i2012

invoke.cont.i2010:                                ; preds = %invoke.cont110
  br i1 undef, label %invoke.cont117, label %if.then.i2011

if.then.i2011:                                    ; preds = %invoke.cont.i2010
  br label %invoke.cont117

lpad.i2012:                                       ; preds = %invoke.cont110
  %tmp98 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont117:                                   ; preds = %if.then.i2011, %invoke.cont.i2010
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2022 unwind label %lpad156.body

lpad:                                             ; preds = %entry
  %tmp118 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup

lpad3:                                            ; preds = %land.rhs, %invoke.cont
  %tmp119 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup

ehcleanup:                                        ; preds = %lpad3, %lpad
  unreachable

lpad16:                                           ; preds = %invoke.cont8
  %tmp121 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup26

lpad20:                                           ; preds = %invoke.cont17
  %tmp122 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup26

ehcleanup26:                                      ; preds = %lpad20, %lpad16
  unreachable

lpad35:                                           ; preds = %land.rhs39, %invoke.cont24
  %tmp124 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad51:                                           ; preds = %invoke.cont44
  %tmp125 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad61:                                           ; preds = %land.rhs58
  %tmp127 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad66.body.thread:                               ; preds = %invoke.cont62
  %tmp128 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad66.body:                                      ; preds = %land.end70
  %tmp129 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad94:                                           ; preds = %invoke.cont95, %invoke.cont91
  %tmp133 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup102

ehcleanup102:                                     ; preds = %lpad94, %lpad.i2000
  unreachable

lpad109:                                          ; preds = %invoke.cont100
  %tmp134 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont.i2022:                                ; preds = %invoke.cont117
  br i1 undef, label %invoke.cont157, label %if.then.i2023

if.then.i2023:                                    ; preds = %invoke.cont.i2022
  br label %invoke.cont157

invoke.cont157:                                   ; preds = %if.then.i2023, %invoke.cont.i2022
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2028 unwind label %lpad164.body

invoke.cont.i2028:                                ; preds = %invoke.cont157
  br i1 undef, label %invoke.cont165, label %if.then.i2029

if.then.i2029:                                    ; preds = %invoke.cont.i2028
  br label %invoke.cont165

invoke.cont165:                                   ; preds = %if.then.i2029, %invoke.cont.i2028
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, void (i8*, i8*)*)*)(i8* undef, i8* undef, void (i8*, i8*)* undef)
          to label %invoke.cont184 unwind label %lpad183

invoke.cont184:                                   ; preds = %invoke.cont165
  %call186 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont185 unwind label %lpad183

invoke.cont185:                                   ; preds = %invoke.cont184
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2034 unwind label %lpad.i2036

invoke.cont.i2034:                                ; preds = %invoke.cont185
  br i1 undef, label %invoke.cont190, label %if.then.i2035

if.then.i2035:                                    ; preds = %invoke.cont.i2034
  br label %invoke.cont190

lpad.i2036:                                       ; preds = %invoke.cont185
  %tmp168 = landingpad { i8*, i32 }
          cleanup
  br label %lpad183.body

invoke.cont190:                                   ; preds = %if.then.i2035, %invoke.cont.i2034
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont197 unwind label %lpad196

invoke.cont197:                                   ; preds = %invoke.cont190
  %call202 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont201 unwind label %lpad200

invoke.cont201:                                   ; preds = %invoke.cont197
  %call205 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont204 unwind label %lpad203

invoke.cont204:                                   ; preds = %invoke.cont201
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2040 unwind label %lpad.i2042

invoke.cont.i2040:                                ; preds = %invoke.cont204
  br i1 undef, label %invoke.cont207, label %if.then.i2041

if.then.i2041:                                    ; preds = %invoke.cont.i2040
  br label %invoke.cont207

lpad.i2042:                                       ; preds = %invoke.cont204
  %tmp181 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont207:                                   ; preds = %if.then.i2041, %invoke.cont.i2040
  %call209 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont208 unwind label %lpad203

invoke.cont208:                                   ; preds = %invoke.cont207
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2046 unwind label %lpad212.body

invoke.cont.i2046:                                ; preds = %invoke.cont208
  br i1 undef, label %invoke.cont213, label %if.then.i2047

if.then.i2047:                                    ; preds = %invoke.cont.i2046
  br label %invoke.cont213

invoke.cont213:                                   ; preds = %if.then.i2047, %invoke.cont.i2046
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont221 unwind label %lpad220

invoke.cont221:                                   ; preds = %invoke.cont213
  %call229 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont228 unwind label %lpad227

invoke.cont228:                                   ; preds = %invoke.cont221
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2052 unwind label %lpad.i2054

invoke.cont.i2052:                                ; preds = %invoke.cont228
  br i1 undef, label %invoke.cont231, label %if.then.i2053

if.then.i2053:                                    ; preds = %invoke.cont.i2052
  br label %invoke.cont231

lpad.i2054:                                       ; preds = %invoke.cont228
  %tmp198 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont231:                                   ; preds = %if.then.i2053, %invoke.cont.i2052
  %call233 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont232 unwind label %lpad227

invoke.cont232:                                   ; preds = %invoke.cont231
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2058 unwind label %lpad236.body

invoke.cont.i2058:                                ; preds = %invoke.cont232
  br i1 undef, label %invoke.cont237, label %if.then.i2059

if.then.i2059:                                    ; preds = %invoke.cont.i2058
  br label %invoke.cont237

invoke.cont237:                                   ; preds = %if.then.i2059, %invoke.cont.i2058
  %call246 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont245 unwind label %lpad244

invoke.cont245:                                   ; preds = %invoke.cont237
  %call248 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 13)
          to label %invoke.cont247 unwind label %lpad244

invoke.cont247:                                   ; preds = %invoke.cont245
  %call251 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 2)
          to label %invoke.cont250 unwind label %lpad249

invoke.cont250:                                   ; preds = %invoke.cont247
  %call254 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 7)
          to label %invoke.cont253 unwind label %lpad252

invoke.cont253:                                   ; preds = %invoke.cont250
  %call257 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8**, i32)*)(i8* undef, i8* undef, i8** undef, i32 3)
          to label %invoke.cont256 unwind label %lpad255

invoke.cont256:                                   ; preds = %invoke.cont253
  %call260 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef, i8* undef)
          to label %invoke.cont259 unwind label %lpad258

invoke.cont259:                                   ; preds = %invoke.cont256
  %call267 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont266 unwind label %lpad265

invoke.cont266:                                   ; preds = %invoke.cont259
  %call275 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef)
          to label %invoke.cont274 unwind label %lpad273

invoke.cont274:                                   ; preds = %invoke.cont266
  %call279 = invoke i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont278 unwind label %lpad277

invoke.cont278:                                   ; preds = %invoke.cont274
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2064 unwind label %lpad.i2066

invoke.cont.i2064:                                ; preds = %invoke.cont278
  br i1 undef, label %invoke.cont281, label %if.then.i2065

if.then.i2065:                                    ; preds = %invoke.cont.i2064
  br label %invoke.cont281

lpad.i2066:                                       ; preds = %invoke.cont278
  %tmp253 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont281:                                   ; preds = %if.then.i2065, %invoke.cont.i2064
  %call291 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont290 unwind label %lpad289

invoke.cont290:                                   ; preds = %invoke.cont281
  %call303 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 8)
          to label %invoke.cont302 unwind label %lpad301

invoke.cont302:                                   ; preds = %invoke.cont290
  %call310 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, double)*)(i8* undef, i8* undef, double 5.000000e-01)
          to label %invoke.cont309 unwind label %lpad308

invoke.cont309:                                   ; preds = %invoke.cont302
  %call313 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 42)
          to label %invoke.cont312 unwind label %lpad311

invoke.cont312:                                   ; preds = %invoke.cont309
  %call316 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8**, i8**, i32)*)(i8* undef, i8* undef, i8** undef, i8** undef, i32 2)
          to label %invoke.cont315 unwind label %lpad314

invoke.cont315:                                   ; preds = %invoke.cont312
  %call322 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef)
          to label %invoke.cont321 unwind label %lpad320

invoke.cont321:                                   ; preds = %invoke.cont315
  br i1 undef, label %land.end344, label %land.rhs335

land.rhs335:                                      ; preds = %invoke.cont321
  %call342 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %land.end344 unwind label %lpad340.body.thread

land.end344:                                      ; preds = %land.rhs335, %invoke.cont321
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2070 unwind label %lpad340.body

invoke.cont.i2070:                                ; preds = %land.end344
  br i1 undef, label %invoke.cont345, label %if.then.i2071

if.then.i2071:                                    ; preds = %invoke.cont.i2070
  br label %invoke.cont345

invoke.cont345:                                   ; preds = %if.then.i2071, %invoke.cont.i2070
  %call362 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef)
          to label %invoke.cont361 unwind label %lpad360

invoke.cont361:                                   ; preds = %invoke.cont345
  %call365 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont364 unwind label %lpad363

invoke.cont364:                                   ; preds = %invoke.cont361
  %call371 = invoke i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont370 unwind label %lpad369

invoke.cont370:                                   ; preds = %invoke.cont364
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2076 unwind label %lpad.i2078

invoke.cont.i2076:                                ; preds = %invoke.cont370
  br i1 undef, label %invoke.cont373, label %if.then.i2077

if.then.i2077:                                    ; preds = %invoke.cont.i2076
  br label %invoke.cont373

lpad.i2078:                                       ; preds = %invoke.cont370
  %tmp340 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont373:                                   ; preds = %if.then.i2077, %invoke.cont.i2076
  %call377 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32, i8*)*)(i8* undef, i8* undef, i32 42, i8* undef)
          to label %invoke.cont376 unwind label %lpad363

invoke.cont376:                                   ; preds = %invoke.cont373
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i32)*)(i8* undef, i8* undef, i8* undef, i32 5)
          to label %invoke.cont382 unwind label %lpad381

invoke.cont382:                                   ; preds = %invoke.cont376
  %call384 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont383 unwind label %lpad381

invoke.cont383:                                   ; preds = %invoke.cont382
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2082 unwind label %lpad.i2084

invoke.cont.i2082:                                ; preds = %invoke.cont383
  br i1 undef, label %invoke.cont392, label %if.then.i2083

if.then.i2083:                                    ; preds = %invoke.cont.i2082
  br label %invoke.cont392

lpad.i2084:                                       ; preds = %invoke.cont383
  %tmp360 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont392:                                   ; preds = %if.then.i2083, %invoke.cont.i2082
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i32)*)(i8* undef, i8* undef, i8* undef, i32 -2)
          to label %invoke.cont395 unwind label %lpad381

invoke.cont395:                                   ; preds = %invoke.cont392
  %call397 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont396 unwind label %lpad381

invoke.cont396:                                   ; preds = %invoke.cont395
  %call400 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont399 unwind label %lpad398

invoke.cont399:                                   ; preds = %invoke.cont396
  %call403 = invoke i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont402 unwind label %lpad401

invoke.cont402:                                   ; preds = %invoke.cont399
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2088 unwind label %lpad.i2090

invoke.cont.i2088:                                ; preds = %invoke.cont402
  br i1 undef, label %invoke.cont405, label %if.then.i2089

if.then.i2089:                                    ; preds = %invoke.cont.i2088
  br label %invoke.cont405

lpad.i2090:                                       ; preds = %invoke.cont402
  %tmp370 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont405:                                   ; preds = %if.then.i2089, %invoke.cont.i2088
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i32)*)(i8* undef, i8* undef, i8* undef, i32 -1)
          to label %invoke.cont408 unwind label %lpad381

invoke.cont408:                                   ; preds = %invoke.cont405
  %call410 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont409 unwind label %lpad381

invoke.cont409:                                   ; preds = %invoke.cont408
  %call413 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont412 unwind label %lpad411

invoke.cont412:                                   ; preds = %invoke.cont409
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2094 unwind label %lpad.i2096

invoke.cont.i2094:                                ; preds = %invoke.cont412
  br i1 undef, label %invoke.cont418, label %if.then.i2095

if.then.i2095:                                    ; preds = %invoke.cont.i2094
  br label %invoke.cont418

lpad.i2096:                                       ; preds = %invoke.cont412
  %tmp380 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont418:                                   ; preds = %if.then.i2095, %invoke.cont.i2094
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i32)*)(i8* undef, i8* undef, i8* undef, i32 0)
          to label %invoke.cont422 unwind label %lpad381

invoke.cont422:                                   ; preds = %invoke.cont418
  %call424 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont423 unwind label %lpad381

invoke.cont423:                                   ; preds = %invoke.cont422
  %call427 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont426 unwind label %lpad425

invoke.cont426:                                   ; preds = %invoke.cont423
  %call430 = invoke i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont429 unwind label %lpad428

invoke.cont429:                                   ; preds = %invoke.cont426
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2100 unwind label %lpad.i2102

invoke.cont.i2100:                                ; preds = %invoke.cont429
  br i1 undef, label %invoke.cont432, label %if.then.i2101

if.then.i2101:                                    ; preds = %invoke.cont.i2100
  br label %invoke.cont432

lpad.i2102:                                       ; preds = %invoke.cont429
  %tmp390 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont432:                                   ; preds = %if.then.i2101, %invoke.cont.i2100
  %call436 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 0)
          to label %invoke.cont435 unwind label %lpad381

invoke.cont435:                                   ; preds = %invoke.cont432
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2106 unwind label %lpad.i2108

invoke.cont.i2106:                                ; preds = %invoke.cont435
  %call444 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 5)
          to label %invoke.cont443 unwind label %lpad381

lpad.i2108:                                       ; preds = %invoke.cont435
  %tmp396 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont443:                                   ; preds = %invoke.cont.i2106
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2112 unwind label %lpad.i2114

invoke.cont.i2112:                                ; preds = %invoke.cont443
  br i1 undef, label %invoke.cont449, label %if.then.i2113

if.then.i2113:                                    ; preds = %invoke.cont.i2112
  br label %invoke.cont449

lpad.i2114:                                       ; preds = %invoke.cont443
  %tmp402 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont449:                                   ; preds = %if.then.i2113, %invoke.cont.i2112
  %call453 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 -2)
          to label %invoke.cont452 unwind label %lpad381

invoke.cont452:                                   ; preds = %invoke.cont449
  %call456 = invoke i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont455 unwind label %lpad454

invoke.cont455:                                   ; preds = %invoke.cont452
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2118 unwind label %lpad.i2120

invoke.cont.i2118:                                ; preds = %invoke.cont455
  br i1 undef, label %invoke.cont458, label %if.then.i2119

if.then.i2119:                                    ; preds = %invoke.cont.i2118
  br label %invoke.cont458

lpad.i2120:                                       ; preds = %invoke.cont455
  %tmp408 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont458:                                   ; preds = %if.then.i2119, %invoke.cont.i2118
  %call461 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 -1)
          to label %invoke.cont460 unwind label %lpad381

invoke.cont460:                                   ; preds = %invoke.cont458
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2124 unwind label %lpad.i2126

invoke.cont.i2124:                                ; preds = %invoke.cont460
  br i1 undef, label %invoke.cont466, label %if.then.i2125

if.then.i2125:                                    ; preds = %invoke.cont.i2124
  br label %invoke.cont466

lpad.i2126:                                       ; preds = %invoke.cont460
  %tmp414 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup477

invoke.cont466:                                   ; preds = %if.then.i2125, %invoke.cont.i2124
  %call470 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 0)
          to label %invoke.cont469 unwind label %lpad381

invoke.cont469:                                   ; preds = %invoke.cont466
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2130 unwind label %lpad.i2132

invoke.cont.i2130:                                ; preds = %invoke.cont469
  br i1 undef, label %invoke.cont475, label %if.then.i2131

if.then.i2131:                                    ; preds = %invoke.cont.i2130
  br label %invoke.cont475

lpad.i2132:                                       ; preds = %invoke.cont469
  %tmp420 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup477

invoke.cont475:                                   ; preds = %if.then.i2131, %invoke.cont.i2130
  %call491 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 1)
          to label %invoke.cont490 unwind label %lpad489

invoke.cont490:                                   ; preds = %invoke.cont475
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont499 unwind label %lpad498

invoke.cont499:                                   ; preds = %invoke.cont490
  %call504 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont503 unwind label %lpad489

invoke.cont503:                                   ; preds = %invoke.cont499
  %call507 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* undef, i8* undef, i32 3)
          to label %invoke.cont506 unwind label %lpad505

invoke.cont506:                                   ; preds = %invoke.cont503
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont509 unwind label %lpad508

invoke.cont509:                                   ; preds = %invoke.cont506
  %call513 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont512 unwind label %lpad489

invoke.cont512:                                   ; preds = %invoke.cont509
  br i1 undef, label %msgSend.null-receiver, label %msgSend.call

msgSend.call:                                     ; preds = %invoke.cont512
  invoke void bitcast (void (i8*, i8*, ...)* @objc_msgSend_stret to void (%struct.CGPoint*, i8*, i8*)*)(%struct.CGPoint* sret(%struct.CGPoint) undef, i8* undef, i8* undef)
          to label %msgSend.cont unwind label %lpad514

msgSend.null-receiver:                            ; preds = %invoke.cont512
  br label %msgSend.cont

msgSend.cont:                                     ; preds = %msgSend.null-receiver, %msgSend.call
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2136 unwind label %lpad.i2138

invoke.cont.i2136:                                ; preds = %msgSend.cont
  br i1 undef, label %invoke.cont521, label %if.then.i2137

if.then.i2137:                                    ; preds = %invoke.cont.i2136
  br label %invoke.cont521

lpad.i2138:                                       ; preds = %msgSend.cont
  %tmp468 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont521:                                   ; preds = %if.then.i2137, %invoke.cont.i2136
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef)
          to label %invoke.cont528 unwind label %lpad527

invoke.cont528:                                   ; preds = %invoke.cont521
  %call532 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont531 unwind label %lpad489

invoke.cont531:                                   ; preds = %invoke.cont528
  %call535 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont534 unwind label %lpad533

invoke.cont534:                                   ; preds = %invoke.cont531
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2142 unwind label %lpad.i2144

invoke.cont.i2142:                                ; preds = %invoke.cont534
  br i1 undef, label %invoke.cont540, label %if.then.i2143

if.then.i2143:                                    ; preds = %invoke.cont.i2142
  br label %invoke.cont540

lpad.i2144:                                       ; preds = %invoke.cont534
  %tmp486 = landingpad { i8*, i32 }
          cleanup
  unreachable

invoke.cont540:                                   ; preds = %if.then.i2143, %invoke.cont.i2142
  %call544 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i32)*)(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef, i32 3)
          to label %invoke.cont543 unwind label %lpad489

invoke.cont543:                                   ; preds = %invoke.cont540
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* undef)
          to label %invoke.cont546 unwind label %lpad545

invoke.cont546:                                   ; preds = %invoke.cont543
  %call549 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont548 unwind label %lpad489

invoke.cont548:                                   ; preds = %invoke.cont546
  %call555 = invoke signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont554 unwind label %lpad553

invoke.cont554:                                   ; preds = %invoke.cont548
  %tmp499 = call i8* @llvm.objc.retain(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*)) #3
  invoke void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i8* %tmp499, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont.i2148 unwind label %lpad.i2150

invoke.cont.i2148:                                ; preds = %invoke.cont554
  call void @llvm.objc.release(i8* %tmp499) #3, !clang.imprecise_release !0
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont566 unwind label %lpad565

lpad.i2150:                                       ; preds = %invoke.cont554
  %tmp500 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.objc.release(i8* %tmp499) #3, !clang.imprecise_release !0
  unreachable

invoke.cont566:                                   ; preds = %invoke.cont.i2148
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, i8*)*)(i8* undef, i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*))
          to label %invoke.cont572 unwind label %lpad571

invoke.cont572:                                   ; preds = %invoke.cont566
  %call582 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont581 unwind label %lpad580

invoke.cont581:                                   ; preds = %invoke.cont572
  unreachable

lpad156.body:                                     ; preds = %invoke.cont117
  %tmp1157 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad164.body:                                     ; preds = %invoke.cont157
  %tmp1158 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad183:                                          ; preds = %invoke.cont184, %invoke.cont165
  %tmp1159 = landingpad { i8*, i32 }
          cleanup
  br label %lpad183.body

lpad183.body:                                     ; preds = %lpad183, %lpad.i2036
  unreachable

lpad196:                                          ; preds = %invoke.cont190
  %tmp1160 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad200:                                          ; preds = %invoke.cont197
  %tmp1161 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad203:                                          ; preds = %invoke.cont207, %invoke.cont201
  %tmp1162 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad212.body:                                     ; preds = %invoke.cont208
  %tmp1163 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad220:                                          ; preds = %invoke.cont213
  %tmp1164 = landingpad { i8*, i32 }
          cleanup
  br label %eh.resume

lpad227:                                          ; preds = %invoke.cont231, %invoke.cont221
  %tmp1166 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup239

lpad236.body:                                     ; preds = %invoke.cont232
  %tmp1167 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup239

ehcleanup239:                                     ; preds = %lpad236.body, %lpad227
  unreachable

lpad244:                                          ; preds = %invoke.cont245, %invoke.cont237
  %tmp1168 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad249:                                          ; preds = %invoke.cont247
  %tmp1169 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad252:                                          ; preds = %invoke.cont250
  %tmp1170 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup263

lpad255:                                          ; preds = %invoke.cont253
  %tmp1171 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup263

lpad258:                                          ; preds = %invoke.cont256
  %tmp1172 = landingpad { i8*, i32 }
          cleanup
  unreachable

ehcleanup263:                                     ; preds = %lpad255, %lpad252
  unreachable

lpad265:                                          ; preds = %invoke.cont259
  %tmp1173 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad273:                                          ; preds = %invoke.cont266
  %tmp1175 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad277:                                          ; preds = %invoke.cont274
  %tmp1176 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad289:                                          ; preds = %invoke.cont281
  %tmp1177 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad301:                                          ; preds = %invoke.cont290
  %tmp1180 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad308:                                          ; preds = %invoke.cont302
  %tmp1182 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad311:                                          ; preds = %invoke.cont309
  %tmp1183 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad314:                                          ; preds = %invoke.cont312
  %tmp1184 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad320:                                          ; preds = %invoke.cont315
  %tmp1186 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad340.body.thread:                              ; preds = %land.rhs335
  %tmp1188 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad340.body:                                     ; preds = %land.end344
  %tmp1189 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad360:                                          ; preds = %invoke.cont345
  %tmp1191 = landingpad { i8*, i32 }
          cleanup
  br label %eh.resume

lpad363:                                          ; preds = %invoke.cont373, %invoke.cont361
  %tmp1192 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad369:                                          ; preds = %invoke.cont364
  %tmp1194 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad381:                                          ; preds = %invoke.cont466, %invoke.cont458, %invoke.cont449, %invoke.cont.i2106, %invoke.cont432, %invoke.cont422, %invoke.cont418, %invoke.cont408, %invoke.cont405, %invoke.cont395, %invoke.cont392, %invoke.cont382, %invoke.cont376
  %tmp1196 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup477

lpad398:                                          ; preds = %invoke.cont396
  %tmp1199 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad401:                                          ; preds = %invoke.cont399
  %tmp1200 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad411:                                          ; preds = %invoke.cont409
  %tmp1201 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad425:                                          ; preds = %invoke.cont423
  %tmp1203 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup477

lpad428:                                          ; preds = %invoke.cont426
  %tmp1204 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad454:                                          ; preds = %invoke.cont452
  %tmp1207 = landingpad { i8*, i32 }
          cleanup
  unreachable

ehcleanup477:                                     ; preds = %lpad425, %lpad381, %lpad.i2132, %lpad.i2126
  unreachable

lpad489:                                          ; preds = %invoke.cont546, %invoke.cont540, %invoke.cont528, %invoke.cont509, %invoke.cont499, %invoke.cont475
  %tmp1211 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup560

lpad498:                                          ; preds = %invoke.cont490
  %tmp1214 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad505:                                          ; preds = %invoke.cont503
  %tmp1215 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad508:                                          ; preds = %invoke.cont506
  %tmp1216 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad514:                                          ; preds = %msgSend.call
  %tmp1217 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad527:                                          ; preds = %invoke.cont521
  %tmp1219 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup560

lpad533:                                          ; preds = %invoke.cont531
  %tmp1220 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad545:                                          ; preds = %invoke.cont543
  %tmp1222 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad553:                                          ; preds = %invoke.cont548
  %tmp1224 = landingpad { i8*, i32 }
          cleanup
  unreachable

ehcleanup560:                                     ; preds = %lpad527, %lpad489
  br label %eh.resume

lpad565:                                          ; preds = %invoke.cont.i2148
  %tmp1225 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad571:                                          ; preds = %invoke.cont566
  %tmp1227 = landingpad { i8*, i32 }
          cleanup
  unreachable

lpad580:                                          ; preds = %invoke.cont572
  %tmp1228 = landingpad { i8*, i32 }
          cleanup
  br label %eh.resume

eh.resume:                                        ; preds = %lpad580, %ehcleanup560, %lpad360, %lpad220
  resume { i8*, i32 } undef
}

@"OBJC_EHTYPE_$_NSException" = external global i8

define void @test4() personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*) {
entry:
  br i1 undef, label %if.end13, label %if.then10

if.then10:                                        ; preds = %entry
  br label %if.end13

if.end13:                                         ; preds = %if.then10, %entry
  %0 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, i8*, i64, i8*, i8)*)(i8* undef, i8* undef, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring to i8*), i64 2, i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring_2 to i8*), i8 signext 0), !clang.arc.no_objc_arc_exceptions !0
  br i1 undef, label %if.then17, label %if.end18

if.then17:                                        ; preds = %if.end13
  br label %if.end18

if.end18:                                         ; preds = %if.then17, %if.end13
  br i1 undef, label %if.then64, label %if.end73

if.then64:                                        ; preds = %if.end18
  br i1 undef, label %cond.end71, label %cond.true68

cond.true68:                                      ; preds = %if.then64
  br label %cond.end71

cond.end71:                                       ; preds = %cond.true68, %if.then64
  br i1 undef, label %cleanup.action, label %cleanup.done

cleanup.action:                                   ; preds = %cond.end71
  br label %cleanup.done

cleanup.done:                                     ; preds = %cleanup.action, %cond.end71
  br label %if.end73

if.end73:                                         ; preds = %cleanup.done, %if.end18
  br i1 undef, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:                                 ; preds = %if.end73
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:                           ; preds = %forcoll.refetch, %forcoll.loopinit
  br label %forcoll.loopbody

forcoll.loopbody:                                 ; preds = %forcoll.notmutated, %forcoll.loopbody.outer
  br i1 undef, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:                                  ; preds = %forcoll.loopbody
  br label %forcoll.notmutated

forcoll.notmutated:                               ; preds = %forcoll.mutated, %forcoll.loopbody
  br i1 undef, label %forcoll.loopbody, label %forcoll.refetch

forcoll.refetch:                                  ; preds = %forcoll.notmutated
  br i1 undef, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:                                    ; preds = %forcoll.refetch, %if.end73
  br i1 undef, label %if.end85, label %if.then82

if.then82:                                        ; preds = %forcoll.empty
  br label %if.end85

if.end85:                                         ; preds = %if.then82, %forcoll.empty
  br i1 undef, label %if.then87, label %if.end102

if.then87:                                        ; preds = %if.end85
  br i1 undef, label %if.end94, label %if.then91

if.then91:                                        ; preds = %if.then87
  br label %if.end94

if.end94:                                         ; preds = %if.then91, %if.then87
  br i1 undef, label %if.end101, label %if.then98

if.then98:                                        ; preds = %if.end94
  br label %if.end101

if.end101:                                        ; preds = %if.then98, %if.end94
  br label %if.end102

if.end102:                                        ; preds = %if.end101, %if.end85
  br i1 undef, label %do.body113, label %if.then107

if.then107:                                       ; preds = %if.end102
  br label %do.body113

do.body113:                                       ; preds = %if.then107, %if.end102
  br i1 undef, label %if.then116, label %if.end117

if.then116:                                       ; preds = %do.body113
  br label %if.end117

if.end117:                                        ; preds = %if.then116, %do.body113
  br i1 undef, label %if.then125, label %if.end126

if.then125:                                       ; preds = %if.end117
  br label %if.end126

if.end126:                                        ; preds = %if.then125, %if.end117
  br i1 undef, label %do.end166, label %cond.true132

cond.true132:                                     ; preds = %if.end126
  br i1 undef, label %do.body148, label %cond.true151

do.body148:                                       ; preds = %cond.true132
  br i1 undef, label %do.end166, label %cond.true151

cond.true151:                                     ; preds = %do.body148, %cond.true132
  br i1 undef, label %if.then162, label %do.end166

if.then162:                                       ; preds = %cond.true151
  br label %do.end166

do.end166:                                        ; preds = %if.then162, %cond.true151, %do.body148, %if.end126
  br i1 undef, label %if.then304, label %if.then170

if.then170:                                       ; preds = %do.end166
  br i1 undef, label %do.end193, label %cond.true179

cond.true179:                                     ; preds = %if.then170
  br i1 undef, label %if.then190, label %do.end193

if.then190:                                       ; preds = %cond.true179
  br label %do.end193

do.end193:                                        ; preds = %if.then190, %cond.true179, %if.then170
  br i1 undef, label %do.body200, label %do.body283

do.body200:                                       ; preds = %do.end193
  br i1 undef, label %do.end254, label %cond.true203

cond.true203:                                     ; preds = %do.body200
  br i1 undef, label %do.body218, label %cond.true221

do.body218:                                       ; preds = %cond.true203
  br i1 undef, label %do.end254, label %cond.true221

cond.true221:                                     ; preds = %do.body218, %cond.true203
  br i1 undef, label %if.then232, label %do.body236

if.then232:                                       ; preds = %cond.true221
  br label %do.body236

do.body236:                                       ; preds = %if.then232, %cond.true221
  br i1 undef, label %do.end254, label %cond.true239

cond.true239:                                     ; preds = %do.body236
  br i1 undef, label %if.then250, label %do.end254

if.then250:                                       ; preds = %cond.true239
  br label %do.end254

do.end254:                                        ; preds = %if.then250, %cond.true239, %do.body236, %do.body218, %do.body200
  br i1 undef, label %do.end277, label %cond.true263

cond.true263:                                     ; preds = %do.end254
  br i1 undef, label %if.then274, label %do.end277

if.then274:                                       ; preds = %cond.true263
  unreachable

do.end277:                                        ; preds = %cond.true263, %do.end254
  br i1 undef, label %if.then280, label %do.body283

if.then280:                                       ; preds = %do.end277
  br label %do.body283

do.body283:                                       ; preds = %if.then280, %do.end277, %do.end193
  br i1 undef, label %if.end301, label %cond.true286

cond.true286:                                     ; preds = %do.body283
  br i1 undef, label %if.then297, label %if.end301

if.then297:                                       ; preds = %cond.true286
  br label %if.end301

if.end301:                                        ; preds = %if.then297, %cond.true286, %do.body283
  br i1 undef, label %if.then304, label %do.body351

if.then304:                                       ; preds = %if.end301, %do.end166
  br i1 undef, label %do.body309.lr.ph, label %do.body351

do.body309.lr.ph:                                 ; preds = %if.then304
  br label %do.body309

do.body309:                                       ; preds = %for.cond.backedge, %do.body309.lr.ph
  br i1 undef, label %do.end328, label %cond.true312

cond.true312:                                     ; preds = %do.body309
  br i1 undef, label %if.then323, label %do.end328

if.then323:                                       ; preds = %cond.true312
  br label %do.end328

do.end328:                                        ; preds = %if.then323, %cond.true312, %do.body309
  br i1 undef, label %for.cond.backedge, label %cond.true335

for.cond.backedge:                                ; preds = %if.then346, %cond.true335, %do.end328
  br i1 undef, label %do.body309, label %do.body351

cond.true335:                                     ; preds = %do.end328
  br i1 undef, label %if.then346, label %for.cond.backedge

if.then346:                                       ; preds = %cond.true335
  br label %for.cond.backedge

do.body351:                                       ; preds = %for.cond.backedge, %if.then304, %if.end301
  br i1 undef, label %if.then354, label %if.end355

if.then354:                                       ; preds = %do.body351
  br label %if.end355

if.end355:                                        ; preds = %if.then354, %do.body351
  br i1 undef, label %if.else, label %if.then364

if.then364:                                       ; preds = %if.end355
  br label %do.body366

if.else:                                          ; preds = %if.end355
  br label %do.body366

do.body366:                                       ; preds = %if.else, %if.then364
  br i1 undef, label %if.then369, label %if.end377.critedge

if.then369:                                       ; preds = %do.body366
  br label %if.end377

if.end377.critedge:                               ; preds = %do.body366
  br label %if.end377

if.end377:                                        ; preds = %if.end377.critedge, %if.then369
  br i1 undef, label %if.then383, label %if.end392.critedge

if.then383:                                       ; preds = %if.end377
  br label %if.end392

if.end392.critedge:                               ; preds = %if.end377
  br label %if.end392

if.end392:                                        ; preds = %if.end392.critedge, %if.then383
  br i1 undef, label %if.then398, label %if.end399

if.then398:                                       ; preds = %if.end392
  br label %if.end399

if.end399:                                        ; preds = %if.then398, %if.end392
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* undef, i8* undef)
          to label %eh.cont unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

eh.cont:                                          ; preds = %if.end399
  br i1 undef, label %if.then430, label %if.end439.critedge

if.then430:                                       ; preds = %eh.cont
  %1 = call i8* @llvm.objc.retain(i8* %0)
  br label %if.end439

lpad:                                             ; preds = %if.end399
  %2 = landingpad { i8*, i32 }
          catch i8* @"OBJC_EHTYPE_$_NSException"
  unreachable

if.end439.critedge:                               ; preds = %eh.cont
  %3 = call i8* @llvm.objc.retain(i8* %0)
  br label %if.end439

if.end439:                                        ; preds = %if.end439.critedge, %if.then430
  call void @llvm.objc.release(i8* %0), !clang.imprecise_release !0
  unreachable

return:                                           ; No predecessors!
  ret void
}


!0 = !{}
