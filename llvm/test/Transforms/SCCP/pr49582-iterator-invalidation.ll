; RUN: opt < %s -ipsccp -disable-output
; PR49582: This test checks for an iterator invalidation issue, which only gets
; exposed on a large-enough test case. We intentionally do not check the output.

@c = external dso_local global i32*, align 8
@d = external dso_local global i32, align 4

define void @f(i32 %i) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.end628, %entry
  %e.0 = phi i32 [ 1, %entry ], [ %e.15, %if.end628 ]
  %cmp = icmp slt i32 %e.0, %i
  call void @llvm.assume(i1 %cmp)
  %0 = load i32*, i32** @c, align 8
  %tobool = icmp ne i32* %0, null
  br i1 %tobool, label %if.then, label %if.end628

if.then:                                          ; preds = %for.cond
  %1 = load i32, i32* %0, align 4
  %tobool1 = icmp ne i32 %1, 0
  br i1 %tobool1, label %if.then2, label %if.else78

if.then2:                                         ; preds = %if.then
  %add = add nsw i32 %e.0, 1
  %cmp3 = icmp sge i32 %add, %i
  br i1 %cmp3, label %if.then4, label %if.end

if.then4:                                         ; preds = %if.then2
  %idxprom = sext i32 %add to i64
  br label %if.end

if.end:                                           ; preds = %if.then4, %if.then2
  br i1 %cmp3, label %if.then9, label %if.end13

if.then9:                                         ; preds = %if.end
  %idxprom11 = sext i32 %add to i64
  br label %if.end13

if.end13:                                         ; preds = %if.then9, %if.end
  br i1 %cmp3, label %if.then16, label %if.end20

if.then16:                                        ; preds = %if.end13
  %idxprom18 = sext i32 %add to i64
  br label %if.end20

if.end20:                                         ; preds = %if.then16, %if.end13
  %add21 = add nsw i32 %e.0, 3
  %cmp22 = icmp sge i32 %add21, %i
  br i1 %cmp22, label %if.then23, label %if.end25

if.then23:                                        ; preds = %if.end20
  br label %if.end25

if.end25:                                         ; preds = %if.then23, %if.end20
  %e.1 = phi i32 [ %add21, %if.then23 ], [ %e.0, %if.end20 ]
  %cmp26 = icmp sge i32 %e.1, %i
  br i1 %cmp26, label %if.then27, label %if.end28

if.then27:                                        ; preds = %if.end25
  %inc = add nsw i32 %e.1, 1
  br label %if.end28

if.end28:                                         ; preds = %if.then27, %if.end25
  %e.2 = phi i32 [ %inc, %if.then27 ], [ %e.1, %if.end25 ]
  %add29 = add nsw i32 %e.2, 2
  %cmp30 = icmp sge i32 %add29, %i
  br i1 %cmp30, label %if.then31, label %if.end33

if.then31:                                        ; preds = %if.end28
  br label %if.end33

if.end33:                                         ; preds = %if.then31, %if.end28
  %e.3 = phi i32 [ %add29, %if.then31 ], [ %e.2, %if.end28 ]
  %cmp34 = icmp sge i32 %e.3, %i
  br i1 %cmp34, label %if.then35, label %if.end38

if.then35:                                        ; preds = %if.end33
  %idxprom36 = sext i32 %e.3 to i64
  br label %if.end38

if.end38:                                         ; preds = %if.then35, %if.end33
  br i1 %cmp34, label %if.then40, label %if.end43

if.then40:                                        ; preds = %if.end38
  %idxprom41 = sext i32 %e.3 to i64
  br label %if.end43

if.end43:                                         ; preds = %if.then40, %if.end38
  br i1 %cmp34, label %if.then45, label %if.end47

if.then45:                                        ; preds = %if.end43
  %inc46 = add nsw i32 %e.3, 1
  br label %if.end47

if.end47:                                         ; preds = %if.then45, %if.end43
  %e.4 = phi i32 [ %inc46, %if.then45 ], [ %e.3, %if.end43 ]
  %cmp48 = icmp sge i32 %e.4, %i
  br i1 %cmp48, label %if.then49, label %if.end51

if.then49:                                        ; preds = %if.end47
  %inc50 = add nsw i32 %e.4, 1
  br label %if.end51

if.end51:                                         ; preds = %if.then49, %if.end47
  %e.5 = phi i32 [ %inc50, %if.then49 ], [ %e.4, %if.end47 ]
  %2 = load i32*, i32** @c, align 8
  %tobool52 = icmp ne i32* %2, null
  br i1 %tobool52, label %if.then53, label %if.else

if.then53:                                        ; preds = %if.end51
  %cmp54 = icmp sge i32 %e.5, %i
  br i1 %cmp54, label %if.then55, label %if.end628

if.then55:                                        ; preds = %if.then53
  unreachable

if.else:                                          ; preds = %if.end51
  %3 = load i32, i32* @d, align 4
  %tobool57 = icmp ne i32 %3, 0
  br i1 %tobool57, label %if.then58, label %if.else68

if.then58:                                        ; preds = %if.else
  %cmp59 = icmp sge i32 %e.5, %i
  br i1 %cmp59, label %if.then60, label %if.end62

if.then60:                                        ; preds = %if.then58
  %inc61 = add nsw i32 %e.5, 1
  br label %if.end62

if.end62:                                         ; preds = %if.then60, %if.then58
  %e.6 = phi i32 [ %inc61, %if.then60 ], [ %e.5, %if.then58 ]
  %add63 = add nsw i32 %e.6, 1
  %cmp64 = icmp sge i32 %add63, %i
  br i1 %cmp64, label %if.then65, label %if.end628

if.then65:                                        ; preds = %if.end62
  br label %if.end628

if.else68:                                        ; preds = %if.else
  %add69 = add nsw i32 %e.5, 2
  %cmp70 = icmp sge i32 %add69, %i
  br i1 %cmp70, label %if.then71, label %if.end628

if.then71:                                        ; preds = %if.else68
  %idxprom73 = sext i32 %add69 to i64
  br label %if.end628

if.else78:                                        ; preds = %if.then
  %call = call i32 @g()
  %tobool79 = icmp ne i32 %call, 0
  br i1 %tobool79, label %if.then80, label %if.else123

if.then80:                                        ; preds = %if.else78
  %add81 = add nsw i32 %e.0, 3
  %cmp82 = icmp sge i32 %add81, %i
  br i1 %cmp82, label %if.then83, label %if.end87

if.then83:                                        ; preds = %if.then80
  %idxprom85 = sext i32 %add81 to i64
  br label %if.end87

if.end87:                                         ; preds = %if.then83, %if.then80
  br i1 %cmp82, label %if.then90, label %if.end94

if.then90:                                        ; preds = %if.end87
  %idxprom92 = sext i32 %add81 to i64
  br label %if.end94

if.end94:                                         ; preds = %if.then90, %if.end87
  br i1 %cmp82, label %if.then97, label %if.end99

if.then97:                                        ; preds = %if.end94
  br label %if.end99

if.end99:                                         ; preds = %if.then97, %if.end94
  %e.7 = phi i32 [ %add81, %if.then97 ], [ %e.0, %if.end94 ]
  %cmp100 = icmp sge i32 %e.7, %i
  br i1 %cmp100, label %if.then101, label %if.end103

if.then101:                                       ; preds = %if.end99
  %inc102 = add nsw i32 %e.7, 1
  br label %if.end103

if.end103:                                        ; preds = %if.then101, %if.end99
  %e.8 = phi i32 [ %inc102, %if.then101 ], [ %e.7, %if.end99 ]
  %add104 = add nsw i32 %e.8, 1
  %cmp105 = icmp sge i32 %add104, %i
  br i1 %cmp105, label %if.then106, label %if.end108

if.then106:                                       ; preds = %if.end103
  br label %if.end108

if.end108:                                        ; preds = %if.then106, %if.end103
  %e.9 = phi i32 [ %add104, %if.then106 ], [ %e.8, %if.end103 ]
  %cmp109 = icmp sge i32 %e.9, %i
  br i1 %cmp109, label %if.then110, label %if.end113

if.then110:                                       ; preds = %if.end108
  %idxprom111 = sext i32 %e.9 to i64
  br label %if.end113

if.end113:                                        ; preds = %if.then110, %if.end108
  br i1 %cmp109, label %if.then115, label %if.end118

if.then115:                                       ; preds = %if.end113
  %idxprom116 = sext i32 %e.9 to i64
  unreachable

if.end118:                                        ; preds = %if.end113
  br i1 %cmp109, label %if.then120, label %if.end628

if.then120:                                       ; preds = %if.end118
  br label %if.end628

if.else123:                                       ; preds = %if.else78
  %call124 = call i32 @g()
  %tobool125 = icmp ne i32 %call124, 0
  br i1 %tobool125, label %if.then126, label %if.end628

if.then126:                                       ; preds = %if.else123
  %call127 = call i32 @g()
  %tobool128 = icmp ne i32 %call127, 0
  br i1 %tobool128, label %if.then129, label %if.else164

if.then129:                                       ; preds = %if.then126
  %add130 = add nsw i32 %e.0, 1
  %cmp131 = icmp sge i32 %add130, %i
  br i1 %cmp131, label %if.then132, label %if.end134

if.then132:                                       ; preds = %if.then129
  br label %if.end134

if.end134:                                        ; preds = %if.then132, %if.then129
  %e.10 = phi i32 [ %add130, %if.then132 ], [ %e.0, %if.then129 ]
  %cmp135 = icmp sge i32 %e.10, %i
  br i1 %cmp135, label %if.then136, label %if.end139

if.then136:                                       ; preds = %if.end134
  %idxprom137 = sext i32 %e.10 to i64
  br label %if.end139

if.end139:                                        ; preds = %if.then136, %if.end134
  br i1 %cmp135, label %if.then141, label %if.end144

if.then141:                                       ; preds = %if.end139
  %idxprom142 = sext i32 %e.10 to i64
  br label %if.end144

if.end144:                                        ; preds = %if.then141, %if.end139
  br i1 %cmp135, label %if.then146, label %if.end149

if.then146:                                       ; preds = %if.end144
  %idxprom147 = sext i32 %e.10 to i64
  br label %if.end149

if.end149:                                        ; preds = %if.then146, %if.end144
  br i1 %cmp135, label %if.then151, label %if.else154

if.then151:                                       ; preds = %if.end149
  %idxprom152 = sext i32 %e.10 to i64
  br label %if.end160

if.else154:                                       ; preds = %if.end149
  %idxprom157 = sext i32 %e.10 to i64
  br label %if.end160

if.end160:                                        ; preds = %if.else154, %if.then151
  br i1 %cmp135, label %if.then162, label %if.end628

if.then162:                                       ; preds = %if.end160
  unreachable

if.else164:                                       ; preds = %if.then126
  %4 = load i32*, i32** @c, align 8
  %tobool165 = icmp ne i32* %4, null
  br i1 %tobool165, label %if.then166, label %if.else195

if.then166:                                       ; preds = %if.else164
  %add167 = add nsw i32 %e.0, 1
  %cmp168 = icmp sge i32 %add167, %i
  br i1 %cmp168, label %if.then169, label %if.end173

if.then169:                                       ; preds = %if.then166
  %idxprom171 = sext i32 %add167 to i64
  br label %if.end173

if.end173:                                        ; preds = %if.then169, %if.then166
  br i1 %cmp168, label %if.then176, label %if.end180

if.then176:                                       ; preds = %if.end173
  %idxprom178 = sext i32 %add167 to i64
  unreachable

if.end180:                                        ; preds = %if.end173
  br i1 %cmp168, label %if.then183, label %if.end187

if.then183:                                       ; preds = %if.end180
  %idxprom185 = sext i32 %add167 to i64
  unreachable

if.end187:                                        ; preds = %if.end180
  br i1 %cmp168, label %if.then190, label %if.end628

if.then190:                                       ; preds = %if.end187
  br label %if.end628

if.else195:                                       ; preds = %if.else164
  %5 = load i32, i32* @d, align 4
  %tobool196 = icmp ne i32 %5, 0
  br i1 %tobool196, label %if.then197, label %if.else205

if.then197:                                       ; preds = %if.else195
  %add198 = add nsw i32 %e.0, 1
  %cmp199 = icmp sge i32 %add198, %i
  br i1 %cmp199, label %if.then200, label %if.end628

if.then200:                                       ; preds = %if.then197
  %idxprom202 = sext i32 %add198 to i64
  br label %if.end628

if.else205:                                       ; preds = %if.else195
  %call206 = call i32 @h()
  %tobool207 = icmp ne i32 %call206, 0
  br i1 %tobool207, label %if.then208, label %if.else217

if.then208:                                       ; preds = %if.else205
  %add209 = add nsw i32 %e.0, 1
  %cmp210 = icmp sge i32 %add209, %i
  br i1 %cmp210, label %if.then211, label %if.end215

if.then211:                                       ; preds = %if.then208
  %idxprom213 = sext i32 %add209 to i64
  unreachable

if.end215:                                        ; preds = %if.then208
  %6 = zext i32 %add209 to i64
  br label %if.end628

if.else217:                                       ; preds = %if.else205
  %7 = load i32*, i32** @c, align 8
  %tobool218 = icmp ne i32* %7, null
  br i1 %tobool218, label %if.then219, label %if.else227

if.then219:                                       ; preds = %if.else217
  %add220 = add nsw i32 %e.0, 1
  %cmp221 = icmp sge i32 %add220, %i
  br i1 %cmp221, label %if.then222, label %if.end628

if.then222:                                       ; preds = %if.then219
  %idxprom224 = sext i32 %add220 to i64
  br label %if.end628

if.else227:                                       ; preds = %if.else217
  %call228 = call i32 @g()
  %tobool229 = icmp ne i32 %call228, 0
  br i1 %tobool229, label %if.then230, label %if.else245

if.then230:                                       ; preds = %if.else227
  %add231 = add nsw i32 %e.0, 1
  %cmp232 = icmp sge i32 %add231, %i
  br i1 %cmp232, label %if.then233, label %if.end237

if.then233:                                       ; preds = %if.then230
  %idxprom235 = sext i32 %add231 to i64
  br label %if.end237

if.end237:                                        ; preds = %if.then233, %if.then230
  br i1 %cmp232, label %if.then240, label %if.end628

if.then240:                                       ; preds = %if.end237
  %idxprom242 = sext i32 %add231 to i64
  br label %if.end628

if.else245:                                       ; preds = %if.else227
  %8 = load i32*, i32** @c, align 8
  %tobool246 = icmp ne i32* %8, null
  br i1 %tobool246, label %if.then247, label %if.else258

if.then247:                                       ; preds = %if.else245
  %add248 = add nsw i32 %e.0, 1
  %cmp249 = icmp sge i32 %add248, %i
  br i1 %cmp249, label %if.then250, label %if.end254

if.then250:                                       ; preds = %if.then247
  %idxprom252 = sext i32 %add248 to i64
  unreachable

if.end254:                                        ; preds = %if.then247
  %9 = zext i32 %add248 to i64
  br label %if.end628

if.else258:                                       ; preds = %if.else245
  %10 = load i32, i32* @d, align 4
  %tobool259 = icmp ne i32 %10, 0
  br i1 %tobool259, label %if.then260, label %if.else268

if.then260:                                       ; preds = %if.else258
  %add261 = add nsw i32 %e.0, 1
  %cmp262 = icmp sge i32 %add261, %i
  br i1 %cmp262, label %if.then263, label %if.end628

if.then263:                                       ; preds = %if.then260
  %idxprom265 = sext i32 %add261 to i64
  br label %if.end628

if.else268:                                       ; preds = %if.else258
  %call269 = call i32 @h()
  %tobool270 = icmp ne i32 %call269, 0
  br i1 %tobool270, label %if.then271, label %if.else279

if.then271:                                       ; preds = %if.else268
  %add272 = add nsw i32 %e.0, 1
  %cmp273 = icmp sge i32 %add272, %i
  br i1 %cmp273, label %if.then274, label %if.end628

if.then274:                                       ; preds = %if.then271
  %idxprom276 = sext i32 %add272 to i64
  br label %if.end628

if.else279:                                       ; preds = %if.else268
  %11 = load i32*, i32** @c, align 8
  %tobool280 = icmp ne i32* %11, null
  br i1 %tobool280, label %if.then281, label %if.else287

if.then281:                                       ; preds = %if.else279
  %add282 = add nsw i32 %e.0, 2
  %cmp283 = icmp sge i32 %add282, %i
  br i1 %cmp283, label %if.then284, label %if.end628

if.then284:                                       ; preds = %if.then281
  br label %if.end628

if.else287:                                       ; preds = %if.else279
  %call288 = call i32 @g()
  %tobool289 = icmp ne i32 %call288, 0
  br i1 %tobool289, label %if.then290, label %if.else307

if.then290:                                       ; preds = %if.else287
  %12 = load i32*, i32** @c, align 8
  %tobool291 = icmp ne i32* %12, null
  br i1 %tobool291, label %if.then292, label %if.else298

if.then292:                                       ; preds = %if.then290
  %add293 = add nsw i32 %e.0, 3
  %cmp294 = icmp sge i32 %add293, %i
  br i1 %cmp294, label %if.then295, label %if.end628

if.then295:                                       ; preds = %if.then292
  br label %if.end628

if.else298:                                       ; preds = %if.then290
  %add299 = add nsw i32 %e.0, 4
  %cmp300 = icmp sge i32 %add299, %i
  br i1 %cmp300, label %if.then301, label %if.end628

if.then301:                                       ; preds = %if.else298
  %idxprom303 = sext i32 %add299 to i64
  br label %if.end628

if.else307:                                       ; preds = %if.else287
  %13 = load i32*, i32** @c, align 8
  %tobool308 = icmp ne i32* %13, null
  br i1 %tobool308, label %if.then309, label %if.else324

if.then309:                                       ; preds = %if.else307
  %add310 = add nsw i32 %e.0, 1
  %cmp311 = icmp sge i32 %add310, %i
  br i1 %cmp311, label %if.then312, label %if.else316

if.then312:                                       ; preds = %if.then309
  %idxprom314 = sext i32 %add310 to i64
  br label %if.end628

if.else316:                                       ; preds = %if.then309
  br i1 undef, label %if.then318, label %if.end628

if.then318:                                       ; preds = %if.else316
  %idxprom320 = sext i32 %add310 to i64
  br label %if.end628

if.else324:                                       ; preds = %if.else307
  %call325 = call i32 @g()
  %tobool326 = icmp ne i32 %call325, 0
  br i1 %tobool326, label %if.then327, label %if.else475

if.then327:                                       ; preds = %if.else324
  %add328 = add nsw i32 %e.0, 2
  %cmp329 = icmp sge i32 %add328, %i
  br i1 %cmp329, label %if.then330, label %if.end332

if.then330:                                       ; preds = %if.then327
  br label %if.end332

if.end332:                                        ; preds = %if.then330, %if.then327
  %e.11 = phi i32 [ %add328, %if.then330 ], [ %e.0, %if.then327 ]
  %cmp333 = icmp sge i32 %e.11, %i
  br i1 %cmp333, label %if.then334, label %if.end336

if.then334:                                       ; preds = %if.end332
  %inc335 = add nsw i32 %e.11, 1
  br label %if.end336

if.end336:                                        ; preds = %if.then334, %if.end332
  %e.12 = phi i32 [ %inc335, %if.then334 ], [ %e.11, %if.end332 ]
  %cmp337 = icmp sge i32 %e.12, %i
  br i1 %cmp337, label %if.then338, label %if.end340

if.then338:                                       ; preds = %if.end336
  %inc339 = add nsw i32 %e.12, 1
  br label %if.end340

if.end340:                                        ; preds = %if.then338, %if.end336
  %e.13 = phi i32 [ %inc339, %if.then338 ], [ %e.12, %if.end336 ]
  %cmp341 = icmp sge i32 %e.13, %i
  br i1 %cmp341, label %if.then342, label %if.end344

if.then342:                                       ; preds = %if.end340
  %inc343 = add nsw i32 %e.13, 1
  br label %if.end344

if.end344:                                        ; preds = %if.then342, %if.end340
  %e.14 = phi i32 [ %inc343, %if.then342 ], [ %e.13, %if.end340 ]
  %call345 = call i32 @g()
  %tobool346 = icmp ne i32 %call345, 0
  br i1 %tobool346, label %if.then347, label %if.else398

if.then347:                                       ; preds = %if.end344
  %cmp348 = icmp sge i32 %e.14, %i
  br i1 %cmp348, label %if.then349, label %if.end352

if.then349:                                       ; preds = %if.then347
  %idxprom350 = sext i32 %e.14 to i64
  br label %if.end352

if.end352:                                        ; preds = %if.then349, %if.then347
  br i1 %cmp348, label %if.then354, label %if.else357

if.then354:                                       ; preds = %if.end352
  %idxprom355 = sext i32 %e.14 to i64
  br label %if.end361

if.else357:                                       ; preds = %if.end352
  %idxprom359 = sext i32 %e.14 to i64
  br label %if.end361

if.end361:                                        ; preds = %if.else357, %if.then354
  br i1 %cmp348, label %if.then363, label %if.end366

if.then363:                                       ; preds = %if.end361
  %idxprom364 = sext i32 %e.14 to i64
  br label %if.end366

if.end366:                                        ; preds = %if.then363, %if.end361
  br i1 %cmp348, label %if.then368, label %if.end371

if.then368:                                       ; preds = %if.end366
  %idxprom369 = sext i32 %e.14 to i64
  br label %if.end371

if.end371:                                        ; preds = %if.then368, %if.end366
  br i1 %cmp348, label %if.then373, label %if.end376

if.then373:                                       ; preds = %if.end371
  %idxprom374 = sext i32 %e.14 to i64
  br label %if.end376

if.end376:                                        ; preds = %if.then373, %if.end371
  br i1 %cmp348, label %if.then378, label %if.end381

if.then378:                                       ; preds = %if.end376
  %idxprom379 = sext i32 %e.14 to i64
  br label %if.end381

if.end381:                                        ; preds = %if.then378, %if.end376
  br i1 %cmp348, label %if.then383, label %if.else386

if.then383:                                       ; preds = %if.end381
  %idxprom384 = sext i32 %e.14 to i64
  br label %if.end390

if.else386:                                       ; preds = %if.end381
  %idxprom388 = sext i32 %e.14 to i64
  br label %if.end390

if.end390:                                        ; preds = %if.else386, %if.then383
  %add391 = add nsw i32 %e.14, 1
  %cmp392 = icmp sge i32 %add391, %i
  br i1 %cmp392, label %if.then393, label %if.end628

if.then393:                                       ; preds = %if.end390
  %idxprom395 = sext i32 %add391 to i64
  br label %if.end628

if.else398:                                       ; preds = %if.end344
  %call399 = call i32 @h()
  %tobool400 = icmp ne i32 %call399, 0
  br i1 %tobool400, label %if.then401, label %if.else409

if.then401:                                       ; preds = %if.else398
  %add402 = add nsw i32 %e.14, 1
  %cmp403 = icmp sge i32 %add402, %i
  br i1 %cmp403, label %if.then404, label %if.end628

if.then404:                                       ; preds = %if.then401
  %idxprom406 = sext i32 %add402 to i64
  br label %if.end628

if.else409:                                       ; preds = %if.else398
  %call410 = call i32 @h()
  %tobool411 = icmp ne i32 %call410, 0
  br i1 %tobool411, label %if.then412, label %if.else420

if.then412:                                       ; preds = %if.else409
  %add413 = add nsw i32 %e.14, 1
  %cmp414 = icmp sge i32 %add413, %i
  br i1 %cmp414, label %if.then415, label %if.end628

if.then415:                                       ; preds = %if.then412
  %idxprom417 = sext i32 %add413 to i64
  br label %if.end628

if.else420:                                       ; preds = %if.else409
  %call421 = call i32 @h()
  %tobool422 = icmp ne i32 %call421, 0
  br i1 %tobool422, label %if.then423, label %if.else431

if.then423:                                       ; preds = %if.else420
  %add424 = add nsw i32 %e.14, 3
  %cmp425 = icmp sge i32 %add424, %i
  br i1 %cmp425, label %if.then426, label %if.end628

if.then426:                                       ; preds = %if.then423
  %idxprom428 = sext i32 %add424 to i64
  br label %if.end628

if.else431:                                       ; preds = %if.else420
  %call432 = call i32 @h()
  %tobool433 = icmp ne i32 %call432, 0
  br i1 %tobool433, label %if.then434, label %if.else440

if.then434:                                       ; preds = %if.else431
  %add435 = add nsw i32 %e.14, 1
  %cmp436 = icmp sge i32 %add435, %i
  br i1 %cmp436, label %if.then437, label %if.end628

if.then437:                                       ; preds = %if.then434
  br label %if.end628

if.else440:                                       ; preds = %if.else431
  %call441 = call i32 @h()
  %tobool442 = icmp ne i32 %call441, 0
  br i1 %tobool442, label %if.then443, label %if.else451

if.then443:                                       ; preds = %if.else440
  %tobool444 = icmp ne i32 %e.14, 0
  br i1 %tobool444, label %if.then445, label %if.end628

if.then445:                                       ; preds = %if.then443
  %cmp446 = icmp sge i32 %e.14, %i
  br i1 %cmp446, label %if.then447, label %if.end628

if.then447:                                       ; preds = %if.then445
  br label %if.end628

if.else451:                                       ; preds = %if.else440
  %call452 = call i32 @h()
  %tobool453 = icmp ne i32 %call452, 0
  br i1 %tobool453, label %if.then454, label %if.else460

if.then454:                                       ; preds = %if.else451
  %add455 = add nsw i32 %e.14, 1
  %cmp456 = icmp sge i32 %add455, %i
  br i1 %cmp456, label %if.then457, label %if.end628

if.then457:                                       ; preds = %if.then454
  br label %if.end628

if.else460:                                       ; preds = %if.else451
  %add461 = add nsw i32 %e.14, 2
  %cmp462 = icmp sge i32 %add461, %i
  br i1 %cmp462, label %if.then463, label %if.end628

if.then463:                                       ; preds = %if.else460
  %idxprom465 = sext i32 %add461 to i64
  br label %if.end628

if.else475:                                       ; preds = %if.else324
  %call476 = call i32 @g()
  %tobool477 = icmp ne i32 %call476, 0
  br i1 %tobool477, label %if.then478, label %if.else509

if.then478:                                       ; preds = %if.else475
  %call479 = call i32 @h()
  %tobool480 = icmp ne i32 %call479, 0
  br i1 %tobool480, label %if.then481, label %if.else487

if.then481:                                       ; preds = %if.then478
  %add482 = add nsw i32 %e.0, 1
  %cmp483 = icmp sge i32 %add482, %i
  br i1 %cmp483, label %if.then484, label %if.end628

if.then484:                                       ; preds = %if.then481
  br label %if.end628

if.else487:                                       ; preds = %if.then478
  %call488 = call i32 @h()
  %tobool489 = icmp ne i32 %call488, 0
  br i1 %tobool489, label %if.then490, label %if.else496

if.then490:                                       ; preds = %if.else487
  %add491 = add nsw i32 %e.0, 1
  %cmp492 = icmp sge i32 %add491, %i
  br i1 %cmp492, label %if.then493, label %if.end628

if.then493:                                       ; preds = %if.then490
  br label %if.end628

if.else496:                                       ; preds = %if.else487
  %add497 = add nsw i32 %e.0, 1
  %cmp498 = icmp sge i32 %add497, %i
  br i1 %cmp498, label %if.then499, label %if.else501

if.then499:                                       ; preds = %if.else496
  br label %if.end628

if.else501:                                       ; preds = %if.else496
  br i1 undef, label %if.then503, label %if.end628

if.then503:                                       ; preds = %if.else501
  br label %if.end628

if.else509:                                       ; preds = %if.else475
  %call510 = call i32 @g()
  %tobool511 = icmp ne i32 %call510, 0
  br i1 %tobool511, label %if.then512, label %if.else565

if.then512:                                       ; preds = %if.else509
  %add513 = add nsw i32 %e.0, 1
  %cmp514 = icmp sge i32 %add513, %i
  br i1 %cmp514, label %if.then515, label %if.end519

if.then515:                                       ; preds = %if.then512
  %idxprom517 = sext i32 %add513 to i64
  br label %if.end519

if.end519:                                        ; preds = %if.then515, %if.then512
  br i1 %cmp514, label %if.then522, label %if.end526

if.then522:                                       ; preds = %if.end519
  %idxprom524 = sext i32 %add513 to i64
  br label %if.end526

if.end526:                                        ; preds = %if.then522, %if.end519
  br i1 %cmp514, label %if.then529, label %if.end533

if.then529:                                       ; preds = %if.end526
  %idxprom531 = sext i32 %add513 to i64
  br label %if.end533

if.end533:                                        ; preds = %if.then529, %if.end526
  %add534 = add nsw i32 %e.0, 2
  %cmp535 = icmp sge i32 %add534, %i
  br i1 %cmp535, label %if.then536, label %if.end540

if.then536:                                       ; preds = %if.end533
  %idxprom538 = sext i32 %add534 to i64
  br label %if.end540

if.end540:                                        ; preds = %if.then536, %if.end533
  br i1 %cmp535, label %if.then543, label %if.end547

if.then543:                                       ; preds = %if.end540
  %idxprom545 = sext i32 %add534 to i64
  unreachable

if.end547:                                        ; preds = %if.end540
  br i1 %cmp514, label %if.then550, label %if.else554

if.then550:                                       ; preds = %if.end547
  %idxprom552 = sext i32 %add513 to i64
  br label %if.end559

if.else554:                                       ; preds = %if.end547
  %idxprom557 = sext i32 %add513 to i64
  br label %if.end559

if.end559:                                        ; preds = %if.else554, %if.then550
  br i1 %cmp514, label %if.then562, label %if.end628

if.then562:                                       ; preds = %if.end559
  br label %if.end628

if.else565:                                       ; preds = %if.else509
  %call566 = call i32 @g()
  %tobool567 = icmp ne i32 %call566, 0
  br i1 %tobool567, label %if.then568, label %if.else590

if.then568:                                       ; preds = %if.else565
  %add569 = add nsw i32 %e.0, 2
  %cmp570 = icmp sge i32 %add569, %i
  br i1 %cmp570, label %if.then571, label %if.else575

if.then571:                                       ; preds = %if.then568
  %idxprom573 = sext i32 %add569 to i64
  br label %if.end582

if.else575:                                       ; preds = %if.then568
  %idxprom579 = sext i32 %add569 to i64
  br label %if.end582

if.end582:                                        ; preds = %if.else575, %if.then571
  %add583 = add nsw i32 %e.0, 1
  %cmp584 = icmp sge i32 %add583, %i
  br i1 %cmp584, label %if.then585, label %if.end628

if.then585:                                       ; preds = %if.end582
  %idxprom587 = sext i32 %add583 to i64
  br label %if.end628

if.else590:                                       ; preds = %if.else565
  %call591 = call i32 @g()
  %tobool592 = icmp ne i32 %call591, 0
  br i1 %tobool592, label %if.then593, label %if.end628

if.then593:                                       ; preds = %if.else590
  %add594 = add nsw i32 %e.0, 1
  %cmp595 = icmp sge i32 %add594, %i
  br i1 %cmp595, label %if.then596, label %if.else600

if.then596:                                       ; preds = %if.then593
  %idxprom598 = sext i32 %add594 to i64
  br label %if.end628

if.else600:                                       ; preds = %if.then593
  br i1 undef, label %if.then602, label %if.end628

if.then602:                                       ; preds = %if.else600
  %idxprom604 = sext i32 %add594 to i64
  br label %if.end628

if.end628:                                        ; preds = %if.then602, %if.else600, %if.then596, %if.else590, %if.then585, %if.end582, %if.then562, %if.end559, %if.then503, %if.else501, %if.then499, %if.then493, %if.then490, %if.then484, %if.then481, %if.then463, %if.else460, %if.then457, %if.then454, %if.then447, %if.then445, %if.then443, %if.then437, %if.then434, %if.then426, %if.then423, %if.then415, %if.then412, %if.then404, %if.then401, %if.then393, %if.end390, %if.then318, %if.else316, %if.then312, %if.then301, %if.else298, %if.then295, %if.then292, %if.then284, %if.then281, %if.then274, %if.then271, %if.then263, %if.then260, %if.end254, %if.then240, %if.end237, %if.then222, %if.then219, %if.end215, %if.then200, %if.then197, %if.then190, %if.end187, %if.end160, %if.else123, %if.then120, %if.end118, %if.then71, %if.else68, %if.then65, %if.end62, %if.then53, %for.cond
  %e.15 = phi i32 [ %e.5, %if.then53 ], [ %add63, %if.then65 ], [ %e.6, %if.end62 ], [ %e.5, %if.then71 ], [ %e.5, %if.else68 ], [ %e.9, %if.then120 ], [ %e.9, %if.end118 ], [ %e.10, %if.end160 ], [ %e.0, %if.then190 ], [ %e.0, %if.end187 ], [ %e.0, %if.then200 ], [ %e.0, %if.then197 ], [ %e.0, %if.end215 ], [ %e.0, %if.then222 ], [ %e.0, %if.then219 ], [ %e.0, %if.then240 ], [ %e.0, %if.end237 ], [ %e.0, %if.end254 ], [ %e.0, %if.then263 ], [ %e.0, %if.then260 ], [ %e.0, %if.then274 ], [ %e.0, %if.then271 ], [ %add282, %if.then284 ], [ %e.0, %if.then281 ], [ %add293, %if.then295 ], [ %e.0, %if.then292 ], [ %e.0, %if.then301 ], [ %e.0, %if.else298 ], [ %e.0, %if.then312 ], [ %e.0, %if.then318 ], [ %e.0, %if.else316 ], [ %e.14, %if.then393 ], [ %e.14, %if.end390 ], [ %e.14, %if.then404 ], [ %e.14, %if.then401 ], [ %e.14, %if.then415 ], [ %e.14, %if.then412 ], [ %e.14, %if.then426 ], [ %e.14, %if.then423 ], [ %add435, %if.then437 ], [ %e.14, %if.then434 ], [ %e.14, %if.then447 ], [ %e.14, %if.then445 ], [ %e.14, %if.then443 ], [ %add455, %if.then457 ], [ %e.14, %if.then454 ], [ %e.14, %if.then463 ], [ %e.14, %if.else460 ], [ %add482, %if.then484 ], [ %e.0, %if.then481 ], [ %add491, %if.then493 ], [ %e.0, %if.then490 ], [ %add497, %if.then499 ], [ %add497, %if.then503 ], [ %e.0, %if.else501 ], [ %add513, %if.then562 ], [ %e.0, %if.end559 ], [ %e.0, %if.then585 ], [ %e.0, %if.end582 ], [ %e.0, %if.then596 ], [ %e.0, %if.then602 ], [ %e.0, %if.else600 ], [ %e.0, %if.else590 ], [ %e.0, %if.else123 ], [ %e.0, %for.cond ]
  br label %for.cond
}

declare i32 @g()

declare i32 @h()

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef)

