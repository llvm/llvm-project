; The Metal pipeline expands what Metal miscompiles, crashes on or leaves
; undefined (found on hardware, llvm/utils/air-gpu): frem, llvm.round's sign
; of zero, roundeven/nearbyint, signed add/sub with overflow, saturating
; conversions, ldexp, frexp and is.fpclass. The portable passes, whose AIR
; the AGX backend compiles, keep them.
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s
; RUN: llc -mtriple=air %s -o - | FileCheck %s --check-prefixes=LLC,NOLLVM
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE
; RUN: llc -mtriple=air -filetype=obj %s -o /dev/null

; CHECK-LABEL: define float @frem_f32(
; CHECK: %r = call float @air.fmod.f32(float %a, float %b)
; PORTABLE-LABEL: define float @frem_f32(
; PORTABLE: frem float
define float @frem_f32(float %a, float %b) {
  %r = frem float %a, %b
  ret float %r
}

; air.fmod.f16 is wrong for x fmod inf: half goes through float.
; CHECK-LABEL: define <2 x half> @frem_v2f16(
; CHECK: %[[X:.*]] = fpext <2 x half> %a to <2 x float>
; CHECK: %[[Y:.*]] = fpext <2 x half> %b to <2 x float>
; CHECK: %[[R:.*]] = call <2 x float> @air.fmod.v2f32(<2 x float> %[[X]], <2 x float> %[[Y]])
; CHECK: fptrunc <2 x float> %[[R]] to <2 x half>
define <2 x half> @frem_v2f16(<2 x half> %a, <2 x half> %b) {
  %r = frem <2 x half> %a, %b
  ret <2 x half> %r
}

; LLC-LABEL: define <4 x float> @roundeven(
; LLC: call <4 x float> @air.rint.v4f32(<4 x float> %a)
; LLC: call <4 x float> @air.rint.v4f32(<4 x float>
; PORTABLE-LABEL: define <4 x float> @roundeven(
; PORTABLE: @llvm.roundeven.v4f32
; PORTABLE: @llvm.nearbyint.v4f32
define <4 x float> @roundeven(<4 x float> %a) {
  %r = call <4 x float> @llvm.roundeven.v4f32(<4 x float> %a)
  %s = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %r)
  ret <4 x float> %s
}

; CHECK-LABEL: define float @round(
; CHECK: %r = call float @llvm.round.f32(float %a)
; CHECK: call float @llvm.copysign.f32(float %r, float %a)
; LLC-LABEL: define float @round(
; LLC: %r = call float @air.round.f32(float %a)
; LLC: call float @llvm.copysign.f32(float %r, float %a)
; PORTABLE-LABEL: define float @round(
; PORTABLE-NOT: copysign
; PORTABLE: ret float
define float @round(float %a) {
  %r = call float @llvm.round.f32(float %a)
  ret float %r
}

; Overflow when the result's sign differs from both operands' (add), or from
; a's and not b's (sub).
; CHECK-LABEL: define i1 @sadd(
; CHECK: %[[R:.*]] = add i32 %a, %b
; CHECK: %[[XA:.*]] = xor i32 %a, %[[R]]
; CHECK: %[[XB:.*]] = xor i32 %b, %[[R]]
; CHECK: %[[S:.*]] = and i32 %[[XA]], %[[XB]]
; CHECK: %[[O:.*]] = icmp slt i32 %[[S]], 0
; CHECK: store i32 %[[R]], ptr %p
; CHECK: ret i1 %[[O]]
define i1 @sadd(i32 %a, i32 %b, ptr %p) {
  %s = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %v = extractvalue { i32, i1 } %s, 0
  %o = extractvalue { i32, i1 } %s, 1
  store i32 %v, ptr %p
  ret i1 %o
}

; Users other than extractvalue get the struct rebuilt.
; CHECK-LABEL: define { <2 x i64>, <2 x i1> } @ssub(
; CHECK: %[[R:.*]] = sub <2 x i64> %a, %b
; CHECK: %[[XA:.*]] = xor <2 x i64> %a, %[[R]]
; CHECK: %[[XB:.*]] = xor <2 x i64> %a, %b
; CHECK: %[[S:.*]] = and <2 x i64> %[[XA]], %[[XB]]
; CHECK: %[[O:.*]] = icmp slt <2 x i64> %[[S]], zeroinitializer
; CHECK: %[[T:.*]] = insertvalue { <2 x i64>, <2 x i1> } poison, <2 x i64> %[[R]], 0
; CHECK: insertvalue { <2 x i64>, <2 x i1> } %[[T]], <2 x i1> %[[O]], 1
define { <2 x i64>, <2 x i1> } @ssub(<2 x i64> %a, <2 x i64> %b) {
  %s = call { <2 x i64>, <2 x i1> } @llvm.ssub.with.overflow.v2i64(<2 x i64> %a, <2 x i64> %b)
  ret { <2 x i64>, <2 x i1> } %s
}

; Below -2^31 or NaN, above the largest float below 2^31, NaN.
; CHECK-LABEL: define i32 @fptosi_sat(
; CHECK: %[[C:.*]] = fptosi float %a to i32
; CHECK: %[[LO:.*]] = fcmp ult float %a, f0xCF000000
; CHECK: %[[R1:.*]] = select i1 %[[LO]], i32 -2147483648, i32 %[[C]]
; CHECK: %[[HI:.*]] = fcmp ogt float %a, f0x4EFFFFFF
; CHECK: %[[R2:.*]] = select i1 %[[HI]], i32 2147483647, i32 %[[R1]]
; CHECK: %[[NAN:.*]] = fcmp uno float %a, %a
; CHECK: select i1 %[[NAN]], i32 0, i32 %[[R2]]
define i32 @fptosi_sat(float %a) {
  %r = call i32 @llvm.fptosi.sat.i32.f32(float %a)
  ret i32 %r
}

; CHECK-LABEL: define <2 x i32> @fptoui_sat(
; CHECK: fptoui <2 x half> %a to <2 x i32>
; CHECK: fcmp ult <2 x half> %a, zeroinitializer
; CHECK: fcmp ogt <2 x half> %a, splat (half 6.550400e+04)
; CHECK: select <2 x i1> %{{.*}}, <2 x i32> splat (i32 -1)
; CHECK: fcmp uno <2 x half> %a, %a
define <2 x i32> @fptoui_sat(<2 x half> %a) {
  %r = call <2 x i32> @llvm.fptoui.sat.v2i32.v2f16(<2 x half> %a)
  ret <2 x i32> %r
}

; A 64-bit exponent is clamped to 32 bits.
; CHECK-LABEL: define float @ldexp(
; CHECK: %l = call float @air.ldexp.f32(float %a, i32 %k)
; CHECK: %[[A:.*]] = call i64 @llvm.smin.i64(i64 %q, i64 2147483647)
; CHECK: %[[B:.*]] = call i64 @llvm.smax.i64(i64 %[[A]], i64 -2147483648)
; CHECK: %[[T:.*]] = trunc i64 %[[B]] to i32
; CHECK: %m = call float @air.ldexp.f32(float %l, i32 %[[T]])
define float @ldexp(float %a, i32 %k, i64 %q) {
  %l = call float @llvm.ldexp.f32.i32(float %a, i32 %k)
  %m = call float @llvm.ldexp.f32.i64(float %l, i64 %q)
  ret float %m
}

; air.frexp's mantissa for -0 is +0: the sign is copied back.
; CHECK-LABEL: define float @frexp(
; CHECK: %[[E:.*]] = alloca <4 x i32>
; CHECK: %[[M:.*]] = call <4 x float> @air.frexp.v4f32(<4 x float> %a, ptr %[[E]])
; CHECK: %[[X:.*]] = load <4 x i32>, ptr %[[E]]
; CHECK: call <4 x float> @llvm.copysign.v4f32(<4 x float> %[[M]], <4 x float> %a)
; CHECK: store <4 x i32> %[[X]], ptr %p
define float @frexp(<4 x float> %a, ptr %p) {
  %f = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> %a)
  %m = extractvalue { <4 x float>, <4 x i32> } %f, 0
  %e = extractvalue { <4 x float>, <4 x i32> } %f, 1
  store <4 x i32> %e, ptr %p
  %r = extractelement <4 x float> %m, i32 0
  ret float %r
}

; 0x3 (nan): |x| > inf. 0x60 (zero): |x| == 0. 0x90 (subnormal):
; |x| - 1 < 0x7fffff. 0x8 (negative normal): |x| - 0x800000 < 0x7f000000 and
; x < 0.
; CHECK-LABEL: define i1 @is_fpclass(
; CHECK: %[[B0:.*]] = bitcast float %a to i32
; CHECK: %[[ABS0:.*]] = and i32 %[[B0]], 2147483647
; CHECK: icmp ugt i32 %[[ABS0]], 2139095040
; CHECK: %[[B1:.*]] = bitcast float %a to i32
; CHECK: %[[ABS1:.*]] = and i32 %[[B1]], 2147483647
; CHECK: icmp eq i32 %[[ABS1]], 0
; CHECK: %[[B2:.*]] = bitcast float %a to i32
; CHECK: %[[ABS:.*]] = and i32 %[[B2]], 2147483647
; CHECK: %[[S1:.*]] = sub i32 %[[ABS]], 1
; CHECK: icmp ult i32 %[[S1]], 8388607
; CHECK: %[[B:.*]] = bitcast float %a to i32
; CHECK: %[[ABS3:.*]] = and i32 %[[B]], 2147483647
; CHECK: %[[N1:.*]] = sub i32 %[[ABS3]], 8388608
; CHECK: %[[N:.*]] = icmp ult i32 %[[N1]], 2130706432
; CHECK: %[[NEG:.*]] = icmp slt i32 %[[B]], 0
; CHECK: and i1 %[[N]], %[[NEG]]
; CHECK: ret i1
define i1 @is_fpclass(float %a) {
  %n = call i1 @llvm.is.fpclass.f32(float %a, i32 3)
  %z = call i1 @llvm.is.fpclass.f32(float %a, i32 96)
  %s = call i1 @llvm.is.fpclass.f32(float %a, i32 144)
  %m = call i1 @llvm.is.fpclass.f32(float %a, i32 8)
  %x = or i1 %n, %z
  %y = or i1 %s, %m
  %r = or i1 %x, %y
  ret i1 %r
}

; NOLLVM-NOT: @llvm.{{(sadd|ssub|fptosi|fptoui|ldexp|frexp|is.fpclass|roundeven|nearbyint)}}
