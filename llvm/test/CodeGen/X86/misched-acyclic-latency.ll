; Check that checkAcyclicLatency() flags loops whose acyclic critical path
; really does span several iterations, and does not flag loops that are limited
; by throughput instead.
;
; RUN: llc < %s -mtriple=x86_64-- -mcpu=x86-64 -debug-only=machine-scheduler \
; RUN:   -o /dev/null 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- -mcpu=skylake-avx512 \
; RUN:   -debug-only=machine-scheduler -o /dev/null 2>&1 | FileCheck %s
;
; REQUIRES: asserts

; @acyclic_limited is a long dependent chain of vector multiplies with only a
; trivial loop-carried dependency, so its acyclic path genuinely spans several
; iterations and must be flagged.  This also keeps the CHECK-NOT below honest:
; it proves the message is still produced and still spelled this way.

; CHECK: ACYCLIC LATENCY LIMIT

; Everything after this point must not be flagged.  The two loops below cover
; the two ways the old estimate got this wrong.
;
; @throughput_loop is a wide, independent chain of 512-bit adds.  Its
; InFlightCount exceeds the micro-op buffer, but the acyclic path fits inside a
; single iteration's own issue time (NumIters=1), so there is nothing for
; cross-iteration overlap to hide.
;
; @port_bound_loop is six independent chains of 512-bit multiplies.  It is
; bound by the vector ports rather than by the front end, so an iteration takes
; several times longer than its micro-op count suggests.  IterCount must come
; from the region's critical resource; taking it from the micro-op issue count
; understates an iteration and overstates how many must be in flight.

; CHECK-NOT: ACYCLIC LATENCY LIMIT

define void @acyclic_limited(ptr %p, ptr %q, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %g = getelementptr inbounds <16 x float>, ptr %p, i64 %iv
  %v = load <16 x float>, ptr %g
  %c0 = fmul <16 x float> %v, %v
  %c1 = fmul <16 x float> %c0, %c0
  %c2 = fmul <16 x float> %c1, %c1
  %c3 = fmul <16 x float> %c2, %c2
  %c4 = fmul <16 x float> %c3, %c3
  %c5 = fmul <16 x float> %c4, %c4
  %c6 = fmul <16 x float> %c5, %c5
  %c7 = fmul <16 x float> %c6, %c6
  %c8 = fmul <16 x float> %c7, %c7
  %c9 = fmul <16 x float> %c8, %c8
  %c10 = fmul <16 x float> %c9, %c9
  %c11 = fmul <16 x float> %c10, %c10
  %c12 = fmul <16 x float> %c11, %c11
  %c13 = fmul <16 x float> %c12, %c12
  %c14 = fmul <16 x float> %c13, %c13
  %c15 = fmul <16 x float> %c14, %c14
  %c16 = fmul <16 x float> %c15, %c15
  %c17 = fmul <16 x float> %c16, %c16
  %c18 = fmul <16 x float> %c17, %c17
  %c19 = fmul <16 x float> %c18, %c18
  %c20 = fmul <16 x float> %c19, %c19
  %c21 = fmul <16 x float> %c20, %c20
  %c22 = fmul <16 x float> %c21, %c21
  %c23 = fmul <16 x float> %c22, %c22
  %c24 = fmul <16 x float> %c23, %c23
  %c25 = fmul <16 x float> %c24, %c24
  %c26 = fmul <16 x float> %c25, %c25
  %c27 = fmul <16 x float> %c26, %c26
  %c28 = fmul <16 x float> %c27, %c27
  %c29 = fmul <16 x float> %c28, %c28
  %c30 = fmul <16 x float> %c29, %c29
  %c31 = fmul <16 x float> %c30, %c30
  %c32 = fmul <16 x float> %c31, %c31
  %c33 = fmul <16 x float> %c32, %c32
  %c34 = fmul <16 x float> %c33, %c33
  %c35 = fmul <16 x float> %c34, %c34
  %c36 = fmul <16 x float> %c35, %c35
  %c37 = fmul <16 x float> %c36, %c36
  %c38 = fmul <16 x float> %c37, %c37
  %c39 = fmul <16 x float> %c38, %c38
  %c40 = fmul <16 x float> %c39, %c39
  %c41 = fmul <16 x float> %c40, %c40
  %c42 = fmul <16 x float> %c41, %c41
  %c43 = fmul <16 x float> %c42, %c42
  %c44 = fmul <16 x float> %c43, %c43
  %c45 = fmul <16 x float> %c44, %c44
  %c46 = fmul <16 x float> %c45, %c45
  %c47 = fmul <16 x float> %c46, %c46
  store <16 x float> %c47, ptr %q
  %iv.next = add i64 %iv, 1
  %cc = icmp slt i64 %iv.next, %n
  br i1 %cc, label %loop, label %exit

exit:
  ret void
}

define void @throughput_loop(ptr %p, ptr %q, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %a0 = getelementptr inbounds <16 x i32>, ptr %p, i64 0
  %v0 = load <16 x i32>, ptr %a0
  %a1 = getelementptr inbounds <16 x i32>, ptr %p, i64 1
  %v1 = load <16 x i32>, ptr %a1
  %a2 = getelementptr inbounds <16 x i32>, ptr %p, i64 2
  %v2 = load <16 x i32>, ptr %a2
  %a3 = getelementptr inbounds <16 x i32>, ptr %p, i64 3
  %v3 = load <16 x i32>, ptr %a3
  %a4 = getelementptr inbounds <16 x i32>, ptr %p, i64 4
  %v4 = load <16 x i32>, ptr %a4
  %a5 = getelementptr inbounds <16 x i32>, ptr %p, i64 5
  %v5 = load <16 x i32>, ptr %a5
  %a6 = getelementptr inbounds <16 x i32>, ptr %p, i64 6
  %v6 = load <16 x i32>, ptr %a6
  %a7 = getelementptr inbounds <16 x i32>, ptr %p, i64 7
  %v7 = load <16 x i32>, ptr %a7
  %t0 = add <16 x i32> %v0, %v1
  %t1 = add <16 x i32> %v1, %v2
  %t2 = add <16 x i32> %v2, %v3
  %t3 = add <16 x i32> %v3, %v4
  %t4 = add <16 x i32> %v4, %v5
  %t5 = add <16 x i32> %v5, %v6
  %t6 = add <16 x i32> %v6, %v7
  %t7 = add <16 x i32> %v7, %v0
  %t8 = add <16 x i32> %t0, %t1
  %t9 = add <16 x i32> %t1, %t2
  %t10 = add <16 x i32> %t2, %t3
  %t11 = add <16 x i32> %t3, %t4
  %t12 = add <16 x i32> %t4, %t5
  %t13 = add <16 x i32> %t5, %t6
  %t14 = add <16 x i32> %t6, %t7
  %t15 = add <16 x i32> %t7, %t0
  %t16 = add <16 x i32> %t8, %t9
  %t17 = add <16 x i32> %t9, %t10
  %t18 = add <16 x i32> %t10, %t11
  %t19 = add <16 x i32> %t11, %t12
  %t20 = add <16 x i32> %t12, %t13
  %t21 = add <16 x i32> %t13, %t14
  %t22 = add <16 x i32> %t14, %t15
  %t23 = add <16 x i32> %t15, %t8
  %t24 = add <16 x i32> %t16, %t17
  %t25 = add <16 x i32> %t17, %t18
  %t26 = add <16 x i32> %t18, %t19
  %t27 = add <16 x i32> %t19, %t20
  %t28 = add <16 x i32> %t20, %t21
  %t29 = add <16 x i32> %t21, %t22
  %t30 = add <16 x i32> %t22, %t23
  %t31 = add <16 x i32> %t23, %t16
  %t32 = add <16 x i32> %t24, %t25
  %t33 = add <16 x i32> %t25, %t26
  %t34 = add <16 x i32> %t26, %t27
  %t35 = add <16 x i32> %t27, %t28
  %t36 = add <16 x i32> %t28, %t29
  %t37 = add <16 x i32> %t29, %t30
  %t38 = add <16 x i32> %t30, %t31
  %t39 = add <16 x i32> %t31, %t24
  %t40 = add <16 x i32> %t32, %t33
  %t41 = add <16 x i32> %t33, %t34
  %t42 = add <16 x i32> %t34, %t35
  %t43 = add <16 x i32> %t35, %t36
  %t44 = add <16 x i32> %t36, %t37
  %t45 = add <16 x i32> %t37, %t38
  %t46 = add <16 x i32> %t38, %t39
  %t47 = add <16 x i32> %t39, %t32
  %t48 = add <16 x i32> %t40, %t41
  %t49 = add <16 x i32> %t41, %t42
  %t50 = add <16 x i32> %t42, %t43
  %t51 = add <16 x i32> %t43, %t44
  %t52 = add <16 x i32> %t44, %t45
  %t53 = add <16 x i32> %t45, %t46
  %t54 = add <16 x i32> %t46, %t47
  %t55 = add <16 x i32> %t47, %t40
  %t56 = add <16 x i32> %t48, %t49
  %t57 = add <16 x i32> %t49, %t50
  %t58 = add <16 x i32> %t50, %t51
  %t59 = add <16 x i32> %t51, %t52
  %t60 = add <16 x i32> %t52, %t53
  %t61 = add <16 x i32> %t53, %t54
  %t62 = add <16 x i32> %t54, %t55
  %t63 = add <16 x i32> %t55, %t48
  %t64 = add <16 x i32> %t56, %t57
  %t65 = add <16 x i32> %t57, %t58
  %t66 = add <16 x i32> %t58, %t59
  %t67 = add <16 x i32> %t59, %t60
  %t68 = add <16 x i32> %t60, %t61
  %t69 = add <16 x i32> %t61, %t62
  %t70 = add <16 x i32> %t62, %t63
  %t71 = add <16 x i32> %t63, %t56
  %t72 = add <16 x i32> %t64, %t65
  %t73 = add <16 x i32> %t65, %t66
  %t74 = add <16 x i32> %t66, %t67
  %t75 = add <16 x i32> %t67, %t68
  %t76 = add <16 x i32> %t68, %t69
  %t77 = add <16 x i32> %t69, %t70
  %t78 = add <16 x i32> %t70, %t71
  %t79 = add <16 x i32> %t71, %t64
  %t80 = add <16 x i32> %t72, %t73
  %t81 = add <16 x i32> %t73, %t74
  %t82 = add <16 x i32> %t74, %t75
  %t83 = add <16 x i32> %t75, %t76
  %t84 = add <16 x i32> %t76, %t77
  %t85 = add <16 x i32> %t77, %t78
  %t86 = add <16 x i32> %t78, %t79
  %t87 = add <16 x i32> %t79, %t72
  %t88 = add <16 x i32> %t80, %t81
  %t89 = add <16 x i32> %t81, %t82
  %t90 = add <16 x i32> %t82, %t83
  %t91 = add <16 x i32> %t83, %t84
  %t92 = add <16 x i32> %t84, %t85
  %t93 = add <16 x i32> %t85, %t86
  %t94 = add <16 x i32> %t86, %t87
  %t95 = add <16 x i32> %t87, %t80
  %t96 = add <16 x i32> %t88, %t89
  %t97 = add <16 x i32> %t89, %t90
  %t98 = add <16 x i32> %t90, %t91
  %t99 = add <16 x i32> %t91, %t92
  %t100 = add <16 x i32> %t92, %t93
  %t101 = add <16 x i32> %t93, %t94
  %t102 = add <16 x i32> %t94, %t95
  %t103 = add <16 x i32> %t95, %t88
  %t104 = add <16 x i32> %t96, %t97
  %t105 = add <16 x i32> %t97, %t98
  %t106 = add <16 x i32> %t98, %t99
  %t107 = add <16 x i32> %t99, %t100
  %t108 = add <16 x i32> %t100, %t101
  %t109 = add <16 x i32> %t101, %t102
  %t110 = add <16 x i32> %t102, %t103
  %t111 = add <16 x i32> %t103, %t96
  %t112 = add <16 x i32> %t104, %t105
  %t113 = add <16 x i32> %t105, %t106
  %t114 = add <16 x i32> %t106, %t107
  %t115 = add <16 x i32> %t107, %t108
  %t116 = add <16 x i32> %t108, %t109
  %t117 = add <16 x i32> %t109, %t110
  %t118 = add <16 x i32> %t110, %t111
  %t119 = add <16 x i32> %t111, %t104
  %t120 = add <16 x i32> %t112, %t113
  %t121 = add <16 x i32> %t113, %t114
  %t122 = add <16 x i32> %t114, %t115
  %t123 = add <16 x i32> %t115, %t116
  %t124 = add <16 x i32> %t116, %t117
  %t125 = add <16 x i32> %t117, %t118
  %t126 = add <16 x i32> %t118, %t119
  %t127 = add <16 x i32> %t119, %t112
  %t128 = add <16 x i32> %t120, %t121
  %t129 = add <16 x i32> %t121, %t122
  %t130 = add <16 x i32> %t122, %t123
  %t131 = add <16 x i32> %t123, %t124
  %t132 = add <16 x i32> %t124, %t125
  %t133 = add <16 x i32> %t125, %t126
  %t134 = add <16 x i32> %t126, %t127
  %t135 = add <16 x i32> %t127, %t120
  %t136 = add <16 x i32> %t128, %t129
  %t137 = add <16 x i32> %t129, %t130
  %t138 = add <16 x i32> %t130, %t131
  %t139 = add <16 x i32> %t131, %t132
  %t140 = add <16 x i32> %t132, %t133
  %t141 = add <16 x i32> %t133, %t134
  %t142 = add <16 x i32> %t134, %t135
  %t143 = add <16 x i32> %t135, %t128
  %t144 = add <16 x i32> %t136, %t137
  %t145 = add <16 x i32> %t137, %t138
  %t146 = add <16 x i32> %t138, %t139
  %t147 = add <16 x i32> %t139, %t140
  %t148 = add <16 x i32> %t140, %t141
  %t149 = add <16 x i32> %t141, %t142
  %t150 = add <16 x i32> %t142, %t143
  %t151 = add <16 x i32> %t143, %t136
  %t152 = add <16 x i32> %t144, %t145
  %t153 = add <16 x i32> %t145, %t146
  %t154 = add <16 x i32> %t146, %t147
  %t155 = add <16 x i32> %t147, %t148
  %t156 = add <16 x i32> %t148, %t149
  %t157 = add <16 x i32> %t149, %t150
  %t158 = add <16 x i32> %t150, %t151
  %t159 = add <16 x i32> %t151, %t144
  %t160 = add <16 x i32> %t152, %t153
  %t161 = add <16 x i32> %t153, %t154
  %t162 = add <16 x i32> %t154, %t155
  %t163 = add <16 x i32> %t155, %t156
  %t164 = add <16 x i32> %t156, %t157
  %t165 = add <16 x i32> %t157, %t158
  %t166 = add <16 x i32> %t158, %t159
  %t167 = add <16 x i32> %t159, %t152
  %t168 = add <16 x i32> %t160, %t161
  %t169 = add <16 x i32> %t161, %t162
  %t170 = add <16 x i32> %t162, %t163
  %t171 = add <16 x i32> %t163, %t164
  %t172 = add <16 x i32> %t164, %t165
  %t173 = add <16 x i32> %t165, %t166
  %t174 = add <16 x i32> %t166, %t167
  %t175 = add <16 x i32> %t167, %t160
  %t176 = add <16 x i32> %t168, %t169
  %t177 = add <16 x i32> %t169, %t170
  %t178 = add <16 x i32> %t170, %t171
  %t179 = add <16 x i32> %t171, %t172
  %t180 = add <16 x i32> %t172, %t173
  %t181 = add <16 x i32> %t173, %t174
  %t182 = add <16 x i32> %t174, %t175
  %t183 = add <16 x i32> %t175, %t168
  %t184 = add <16 x i32> %t176, %t177
  %t185 = add <16 x i32> %t177, %t178
  %t186 = add <16 x i32> %t178, %t179
  %t187 = add <16 x i32> %t179, %t180
  %t188 = add <16 x i32> %t180, %t181
  %t189 = add <16 x i32> %t181, %t182
  %t190 = add <16 x i32> %t182, %t183
  %t191 = add <16 x i32> %t183, %t176
  %t192 = add <16 x i32> %t184, %t185
  %t193 = add <16 x i32> %t185, %t186
  %t194 = add <16 x i32> %t186, %t187
  %t195 = add <16 x i32> %t187, %t188
  %t196 = add <16 x i32> %t188, %t189
  %t197 = add <16 x i32> %t189, %t190
  %t198 = add <16 x i32> %t190, %t191
  %t199 = add <16 x i32> %t191, %t184
  %t200 = add <16 x i32> %t192, %t193
  %t201 = add <16 x i32> %t193, %t194
  %t202 = add <16 x i32> %t194, %t195
  %t203 = add <16 x i32> %t195, %t196
  %t204 = add <16 x i32> %t196, %t197
  %t205 = add <16 x i32> %t197, %t198
  %t206 = add <16 x i32> %t198, %t199
  %t207 = add <16 x i32> %t199, %t192
  %t208 = add <16 x i32> %t200, %t201
  %t209 = add <16 x i32> %t201, %t202
  %t210 = add <16 x i32> %t202, %t203
  %t211 = add <16 x i32> %t203, %t204
  %t212 = add <16 x i32> %t204, %t205
  %t213 = add <16 x i32> %t205, %t206
  %t214 = add <16 x i32> %t206, %t207
  %t215 = add <16 x i32> %t207, %t200
  %t216 = add <16 x i32> %t208, %t209
  %t217 = add <16 x i32> %t209, %t210
  %t218 = add <16 x i32> %t210, %t211
  %t219 = add <16 x i32> %t211, %t212
  %t220 = add <16 x i32> %t212, %t213
  %t221 = add <16 x i32> %t213, %t214
  %t222 = add <16 x i32> %t214, %t215
  %t223 = add <16 x i32> %t215, %t208
  %t224 = add <16 x i32> %t216, %t217
  %t225 = add <16 x i32> %t217, %t218
  %t226 = add <16 x i32> %t218, %t219
  %t227 = add <16 x i32> %t219, %t220
  %t228 = add <16 x i32> %t220, %t221
  %t229 = add <16 x i32> %t221, %t222
  %t230 = add <16 x i32> %t222, %t223
  %t231 = add <16 x i32> %t223, %t216
  %t232 = add <16 x i32> %t224, %t225
  %t233 = add <16 x i32> %t225, %t226
  %t234 = add <16 x i32> %t226, %t227
  %t235 = add <16 x i32> %t227, %t228
  %t236 = add <16 x i32> %t228, %t229
  %t237 = add <16 x i32> %t229, %t230
  %t238 = add <16 x i32> %t230, %t231
  %t239 = add <16 x i32> %t231, %t224
  %t240 = add <16 x i32> %t232, %t233
  %t241 = add <16 x i32> %t233, %t234
  %t242 = add <16 x i32> %t234, %t235
  %t243 = add <16 x i32> %t235, %t236
  %t244 = add <16 x i32> %t236, %t237
  %t245 = add <16 x i32> %t237, %t238
  %t246 = add <16 x i32> %t238, %t239
  %t247 = add <16 x i32> %t239, %t232
  %t248 = add <16 x i32> %t240, %t241
  %t249 = add <16 x i32> %t241, %t242
  %t250 = add <16 x i32> %t242, %t243
  %t251 = add <16 x i32> %t243, %t244
  %t252 = add <16 x i32> %t244, %t245
  %t253 = add <16 x i32> %t245, %t246
  %t254 = add <16 x i32> %t246, %t247
  %t255 = add <16 x i32> %t247, %t240
  %t256 = add <16 x i32> %t248, %t249
  %t257 = add <16 x i32> %t249, %t250
  %t258 = add <16 x i32> %t250, %t251
  %t259 = add <16 x i32> %t251, %t252
  %t260 = add <16 x i32> %t252, %t253
  %t261 = add <16 x i32> %t253, %t254
  %t262 = add <16 x i32> %t254, %t255
  %t263 = add <16 x i32> %t255, %t248
  %t264 = add <16 x i32> %t256, %t257
  %t265 = add <16 x i32> %t257, %t258
  %t266 = add <16 x i32> %t258, %t259
  %t267 = add <16 x i32> %t259, %t260
  %t268 = add <16 x i32> %t260, %t261
  %t269 = add <16 x i32> %t261, %t262
  %t270 = add <16 x i32> %t262, %t263
  %t271 = add <16 x i32> %t263, %t256
  %t272 = add <16 x i32> %t264, %t265
  %t273 = add <16 x i32> %t265, %t266
  %t274 = add <16 x i32> %t266, %t267
  %t275 = add <16 x i32> %t267, %t268
  %t276 = add <16 x i32> %t268, %t269
  %t277 = add <16 x i32> %t269, %t270
  %t278 = add <16 x i32> %t270, %t271
  %t279 = add <16 x i32> %t271, %t264
  %t280 = add <16 x i32> %t272, %t273
  %t281 = add <16 x i32> %t273, %t274
  %t282 = add <16 x i32> %t274, %t275
  %t283 = add <16 x i32> %t275, %t276
  %t284 = add <16 x i32> %t276, %t277
  %t285 = add <16 x i32> %t277, %t278
  %t286 = add <16 x i32> %t278, %t279
  %t287 = add <16 x i32> %t279, %t272
  %t288 = add <16 x i32> %t280, %t281
  %t289 = add <16 x i32> %t281, %t282
  %t290 = add <16 x i32> %t282, %t283
  %t291 = add <16 x i32> %t283, %t284
  %t292 = add <16 x i32> %t284, %t285
  %t293 = add <16 x i32> %t285, %t286
  %t294 = add <16 x i32> %t286, %t287
  %t295 = add <16 x i32> %t287, %t280
  %t296 = add <16 x i32> %t288, %t289
  %t297 = add <16 x i32> %t289, %t290
  %t298 = add <16 x i32> %t290, %t291
  %t299 = add <16 x i32> %t291, %t292
  %t300 = add <16 x i32> %t292, %t293
  %t301 = add <16 x i32> %t293, %t294
  %t302 = add <16 x i32> %t294, %t295
  %t303 = add <16 x i32> %t295, %t288
  %t304 = add <16 x i32> %t296, %t297
  %t305 = add <16 x i32> %t297, %t298
  %t306 = add <16 x i32> %t298, %t299
  %t307 = add <16 x i32> %t299, %t300
  %t308 = add <16 x i32> %t300, %t301
  %t309 = add <16 x i32> %t301, %t302
  %t310 = add <16 x i32> %t302, %t303
  %t311 = add <16 x i32> %t303, %t296
  %t312 = add <16 x i32> %t304, %t305
  %t313 = add <16 x i32> %t305, %t306
  %t314 = add <16 x i32> %t306, %t307
  %t315 = add <16 x i32> %t307, %t308
  %t316 = add <16 x i32> %t308, %t309
  %t317 = add <16 x i32> %t309, %t310
  %t318 = add <16 x i32> %t310, %t311
  %t319 = add <16 x i32> %t311, %t304
  store <16 x i32> %t312, ptr %q
  %iv.next = add i64 %iv, 1
  %c = icmp slt i64 %iv.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

define void @port_bound_loop(ptr noalias %p, ptr noalias %q, i64 %n) #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %g0 = getelementptr <16 x float>, ptr %p, i64 0
  %v0.0 = load <16 x float>, ptr %g0, align 64
  %g1 = getelementptr <16 x float>, ptr %p, i64 1
  %v1.0 = load <16 x float>, ptr %g1, align 64
  %g2 = getelementptr <16 x float>, ptr %p, i64 2
  %v2.0 = load <16 x float>, ptr %g2, align 64
  %g3 = getelementptr <16 x float>, ptr %p, i64 3
  %v3.0 = load <16 x float>, ptr %g3, align 64
  %g4 = getelementptr <16 x float>, ptr %p, i64 4
  %v4.0 = load <16 x float>, ptr %g4, align 64
  %g5 = getelementptr <16 x float>, ptr %p, i64 5
  %v5.0 = load <16 x float>, ptr %g5, align 64
  %v0.1 = fmul <16 x float> %v0.0, splat (float 1.00e+00)
  %v1.1 = fmul <16 x float> %v1.0, splat (float 1.01e+00)
  %v2.1 = fmul <16 x float> %v2.0, splat (float 1.02e+00)
  %v3.1 = fmul <16 x float> %v3.0, splat (float 1.03e+00)
  %v4.1 = fmul <16 x float> %v4.0, splat (float 1.04e+00)
  %v5.1 = fmul <16 x float> %v5.0, splat (float 1.05e+00)
  %v0.2 = fmul <16 x float> %v0.1, splat (float 1.00e+00)
  %v1.2 = fmul <16 x float> %v1.1, splat (float 1.01e+00)
  %v2.2 = fmul <16 x float> %v2.1, splat (float 1.02e+00)
  %v3.2 = fmul <16 x float> %v3.1, splat (float 1.03e+00)
  %v4.2 = fmul <16 x float> %v4.1, splat (float 1.04e+00)
  %v5.2 = fmul <16 x float> %v5.1, splat (float 1.05e+00)
  %v0.3 = fmul <16 x float> %v0.2, splat (float 1.00e+00)
  %v1.3 = fmul <16 x float> %v1.2, splat (float 1.01e+00)
  %v2.3 = fmul <16 x float> %v2.2, splat (float 1.02e+00)
  %v3.3 = fmul <16 x float> %v3.2, splat (float 1.03e+00)
  %v4.3 = fmul <16 x float> %v4.2, splat (float 1.04e+00)
  %v5.3 = fmul <16 x float> %v5.2, splat (float 1.05e+00)
  %v0.4 = fmul <16 x float> %v0.3, splat (float 1.00e+00)
  %v1.4 = fmul <16 x float> %v1.3, splat (float 1.01e+00)
  %v2.4 = fmul <16 x float> %v2.3, splat (float 1.02e+00)
  %v3.4 = fmul <16 x float> %v3.3, splat (float 1.03e+00)
  %v4.4 = fmul <16 x float> %v4.3, splat (float 1.04e+00)
  %v5.4 = fmul <16 x float> %v5.3, splat (float 1.05e+00)
  %v0.5 = fmul <16 x float> %v0.4, splat (float 1.00e+00)
  %v1.5 = fmul <16 x float> %v1.4, splat (float 1.01e+00)
  %v2.5 = fmul <16 x float> %v2.4, splat (float 1.02e+00)
  %v3.5 = fmul <16 x float> %v3.4, splat (float 1.03e+00)
  %v4.5 = fmul <16 x float> %v4.4, splat (float 1.04e+00)
  %v5.5 = fmul <16 x float> %v5.4, splat (float 1.05e+00)
  %v0.6 = fmul <16 x float> %v0.5, splat (float 1.00e+00)
  %v1.6 = fmul <16 x float> %v1.5, splat (float 1.01e+00)
  %v2.6 = fmul <16 x float> %v2.5, splat (float 1.02e+00)
  %v3.6 = fmul <16 x float> %v3.5, splat (float 1.03e+00)
  %v4.6 = fmul <16 x float> %v4.5, splat (float 1.04e+00)
  %v5.6 = fmul <16 x float> %v5.5, splat (float 1.05e+00)
  %v0.7 = fmul <16 x float> %v0.6, splat (float 1.00e+00)
  %v1.7 = fmul <16 x float> %v1.6, splat (float 1.01e+00)
  %v2.7 = fmul <16 x float> %v2.6, splat (float 1.02e+00)
  %v3.7 = fmul <16 x float> %v3.6, splat (float 1.03e+00)
  %v4.7 = fmul <16 x float> %v4.6, splat (float 1.04e+00)
  %v5.7 = fmul <16 x float> %v5.6, splat (float 1.05e+00)
  %v0.8 = fmul <16 x float> %v0.7, splat (float 1.00e+00)
  %v1.8 = fmul <16 x float> %v1.7, splat (float 1.01e+00)
  %v2.8 = fmul <16 x float> %v2.7, splat (float 1.02e+00)
  %v3.8 = fmul <16 x float> %v3.7, splat (float 1.03e+00)
  %v4.8 = fmul <16 x float> %v4.7, splat (float 1.04e+00)
  %v5.8 = fmul <16 x float> %v5.7, splat (float 1.05e+00)
  %v0.9 = fmul <16 x float> %v0.8, splat (float 1.00e+00)
  %v1.9 = fmul <16 x float> %v1.8, splat (float 1.01e+00)
  %v2.9 = fmul <16 x float> %v2.8, splat (float 1.02e+00)
  %v3.9 = fmul <16 x float> %v3.8, splat (float 1.03e+00)
  %v4.9 = fmul <16 x float> %v4.8, splat (float 1.04e+00)
  %v5.9 = fmul <16 x float> %v5.8, splat (float 1.05e+00)
  %v0.10 = fmul <16 x float> %v0.9, splat (float 1.00e+00)
  %v1.10 = fmul <16 x float> %v1.9, splat (float 1.01e+00)
  %v2.10 = fmul <16 x float> %v2.9, splat (float 1.02e+00)
  %v3.10 = fmul <16 x float> %v3.9, splat (float 1.03e+00)
  %v4.10 = fmul <16 x float> %v4.9, splat (float 1.04e+00)
  %v5.10 = fmul <16 x float> %v5.9, splat (float 1.05e+00)
  %v0.11 = fmul <16 x float> %v0.10, splat (float 1.00e+00)
  %v1.11 = fmul <16 x float> %v1.10, splat (float 1.01e+00)
  %v2.11 = fmul <16 x float> %v2.10, splat (float 1.02e+00)
  %v3.11 = fmul <16 x float> %v3.10, splat (float 1.03e+00)
  %v4.11 = fmul <16 x float> %v4.10, splat (float 1.04e+00)
  %v5.11 = fmul <16 x float> %v5.10, splat (float 1.05e+00)
  %v0.12 = fmul <16 x float> %v0.11, splat (float 1.00e+00)
  %v1.12 = fmul <16 x float> %v1.11, splat (float 1.01e+00)
  %v2.12 = fmul <16 x float> %v2.11, splat (float 1.02e+00)
  %v3.12 = fmul <16 x float> %v3.11, splat (float 1.03e+00)
  %v4.12 = fmul <16 x float> %v4.11, splat (float 1.04e+00)
  %v5.12 = fmul <16 x float> %v5.11, splat (float 1.05e+00)
  %v0.13 = fmul <16 x float> %v0.12, splat (float 1.00e+00)
  %v1.13 = fmul <16 x float> %v1.12, splat (float 1.01e+00)
  %v2.13 = fmul <16 x float> %v2.12, splat (float 1.02e+00)
  %v3.13 = fmul <16 x float> %v3.12, splat (float 1.03e+00)
  %v4.13 = fmul <16 x float> %v4.12, splat (float 1.04e+00)
  %v5.13 = fmul <16 x float> %v5.12, splat (float 1.05e+00)
  %v0.14 = fmul <16 x float> %v0.13, splat (float 1.00e+00)
  %v1.14 = fmul <16 x float> %v1.13, splat (float 1.01e+00)
  %v2.14 = fmul <16 x float> %v2.13, splat (float 1.02e+00)
  %v3.14 = fmul <16 x float> %v3.13, splat (float 1.03e+00)
  %v4.14 = fmul <16 x float> %v4.13, splat (float 1.04e+00)
  %v5.14 = fmul <16 x float> %v5.13, splat (float 1.05e+00)
  %v0.15 = fmul <16 x float> %v0.14, splat (float 1.00e+00)
  %v1.15 = fmul <16 x float> %v1.14, splat (float 1.01e+00)
  %v2.15 = fmul <16 x float> %v2.14, splat (float 1.02e+00)
  %v3.15 = fmul <16 x float> %v3.14, splat (float 1.03e+00)
  %v4.15 = fmul <16 x float> %v4.14, splat (float 1.04e+00)
  %v5.15 = fmul <16 x float> %v5.14, splat (float 1.05e+00)
  %s0 = getelementptr <16 x float>, ptr %q, i64 0
  store <16 x float> %v0.15, ptr %s0, align 64
  %s1 = getelementptr <16 x float>, ptr %q, i64 1
  store <16 x float> %v1.15, ptr %s1, align 64
  %s2 = getelementptr <16 x float>, ptr %q, i64 2
  store <16 x float> %v2.15, ptr %s2, align 64
  %s3 = getelementptr <16 x float>, ptr %q, i64 3
  store <16 x float> %v3.15, ptr %s3, align 64
  %s4 = getelementptr <16 x float>, ptr %q, i64 4
  store <16 x float> %v4.15, ptr %s4, align 64
  %s5 = getelementptr <16 x float>, ptr %q, i64 5
  store <16 x float> %v5.15, ptr %s5, align 64
  %i.next = add i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
attributes #0 = { "target-features"="+avx512f" }
