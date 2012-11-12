; RUN: opt -objc-arc -S < %s
; rdar://12277446

; The total number of paths grows exponentially with the number of branches, and a
; computation of this number can overflow any reasonable fixed-sized integer.

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.0.0"

%struct.NSConstantString.11.33.55.77.99.121.143.332.1130.1340.2768 = type { i32*, i32, i8*, i32 }

@_unnamed_cfstring_591 = external constant %struct.NSConstantString.11.33.55.77.99.121.143.332.1130.1340.2768, section "__DATA,__cfstring"

declare i8* @objc_retain(i8*) nonlazybind

declare void @objc_release(i8*) nonlazybind

define hidden void @foo() {
entry:
  br i1 undef, label %msgSend.nullinit, label %msgSend.call

msgSend.call:                                     ; preds = %entry
  br label %msgSend.cont

msgSend.nullinit:                                 ; preds = %entry
  br label %msgSend.cont

msgSend.cont:                                     ; preds = %msgSend.nullinit, %msgSend.call
  %0 = bitcast %struct.NSConstantString.11.33.55.77.99.121.143.332.1130.1340.2768* @_unnamed_cfstring_591 to i8*
  %1 = call i8* @objc_retain(i8* %0) nounwind
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
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

!0 = metadata !{}
