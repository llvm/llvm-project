; RUN: llc < %s -asm-verbose=false -mtriple=aarch64-none-eabi -mattr=-fullfp16 | FileCheck %s --check-prefix=CHECK-CVT --check-prefix=CHECK-COMMON
; RUN: llc < %s -asm-verbose=false -mtriple=aarch64-none-eabi -mattr=+fullfp16 | FileCheck %s --check-prefix=CHECK-FP16 --check-prefix=CHECK-COMMON

define <4 x half> @add_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-CVT-LABEL: add_h:
; CHECK-CVT-DAG:   fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-CVT-DAG:   fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK-CVT-NEXT:  fadd [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK-CVT-NEXT:  fcvtn v0.4h, [[RES]]

; CHECK-FP16-LABEL: add_h:
; CHECK-FP16:       fadd  v0.4h, v0.4h, v1.4h
; CHECK-FP16-NEXT:  ret
  %0 = fadd <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @build_h4(<4 x half> %a) {
entry:
; CHECK-COMMON-LABEL: build_h4:
; CHECK-COMMON:       mov [[GPR:w[0-9]+]], #15565
; CHECK-COMMON-NEXT:  dup v0.4h, [[GPR]]
  ret <4 x half> <half 0xH3CCD, half 0xH3CCD, half 0xH3CCD, half 0xH3CCD>
}


define <4 x half> @sub_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-CVT-LABEL: sub_h:
; CHECK-CVT-DAG:   fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-CVT-DAG:   fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK-CVT-NEXT:  fsub [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK-CVT-NEXT:  fcvtn v0.4h, [[RES]]

; CHECK-FP16-LABEL: sub_h:
; CHECK-FP16:       fsub  v0.4h, v0.4h, v1.4h
; CHECK-FP16-NEXT:  ret
  %0 = fsub <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @mul_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-CVT-LABEL: mul_h:
; CHECK-CVT-DAG:   fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-CVT-DAG:   fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK-CVT-NEXT:  fmul [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK-CVT-NEXT:  fcvtn v0.4h, [[RES]]

; CHECK-FP16-LABEL: mul_h:
; CHECK-FP16:       fmul  v0.4h, v0.4h, v1.4h
; CHECK-FP16-NEXT:  ret
  %0 = fmul <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @div_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-CVT-LABEL: div_h:
; CHECK-CVT-DAG:   fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-CVT-DAG:   fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK-CVT-NEXT:  fdiv [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK-CVT-NEXT:  fcvtn v0.4h, [[RES]]

; CHECK-FP16-LABEL: div_h:
; CHECK-FP16:       fdiv  v0.4h, v0.4h, v1.4h
; CHECK-FP16-NEXT:  ret
  %0 = fdiv <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @load_h(<4 x half>* %a) {
entry:
; CHECK-COMMON-LABEL: load_h:
; CHECK-COMMON:       ldr d0, [x0]
; CHECK-COMMON-NEXT:  ret
  %0 = load <4 x half>, <4 x half>* %a, align 4
  ret <4 x half> %0
}


define void @store_h(<4 x half>* %a, <4 x half> %b) {
entry:
; CHECK-COMMON-LABEL: store_h:
; CHECK-COMMON:       str d0, [x0]
; CHECK-COMMON-NEXT:  ret
  store <4 x half> %b, <4 x half>* %a, align 4
  ret void
}

define <4 x half> @s_to_h(<4 x float> %a) {
; CHECK-COMMON-LABEL: s_to_h:
; CHECK-COMMON:       fcvtn v0.4h, v0.4s
; CHECK-COMMON-NEXT:  ret
  %1 = fptrunc <4 x float> %a to <4 x half>
  ret <4 x half> %1
}

define <4 x half> @d_to_h(<4 x double> %a) {
; CHECK-LABEL: d_to_h:
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: mov v{{[0-9]+}}.h
; CHECK-DAG: mov v{{[0-9]+}}.h
; CHECK-DAG: mov v{{[0-9]+}}.h
; CHECK-DAG: mov v{{[0-9]+}}.h
  %1 = fptrunc <4 x double> %a to <4 x half>
  ret <4 x half> %1
}

define <4 x float> @h_to_s(<4 x half> %a) {
; CHECK-COMMON-LABEL: h_to_s:
; CHECK-COMMON:       fcvtl v0.4s, v0.4h
; CHECK-COMMON-NEXT:  ret
  %1 = fpext <4 x half> %a to <4 x float>
  ret <4 x float> %1
}

define <4 x double> @h_to_d(<4 x half> %a) {
; CHECK-LABEL: h_to_d:
; CHECK-DAG: mov h{{[0-9]+}}, v0.h
; CHECK-DAG: mov h{{[0-9]+}}, v0.h
; CHECK-DAG: mov h{{[0-9]+}}, v0.h
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
  %1 = fpext <4 x half> %a to <4 x double>
  ret <4 x double> %1
}

define <4 x half> @bitcast_i_to_h(float, <4 x i16> %a) {
; CHECK-COMMON-LABEL: bitcast_i_to_h:
; CHECK-COMMON:       mov v0.16b, v1.16b
; CHECK-COMMON-NEXT:  ret
  %2 = bitcast <4 x i16> %a to <4 x half>
  ret <4 x half> %2
}

define <4 x i16> @bitcast_h_to_i(float, <4 x half> %a) {
; CHECK-COMMON-LABEL: bitcast_h_to_i:
; CHECK-COMMON:       mov v0.16b, v1.16b
; CHECK-COMMON-NEXT:  ret
  %2 = bitcast <4 x half> %a to <4 x i16>
  ret <4 x i16> %2
}

define <4 x half> @sitofp_i8(<4 x i8> %a) #0 {
; CHECK-COMMON-LABEL: sitofp_i8:
; CHECK-COMMON-NEXT:  shl [[OP1:v[0-9]+\.4h]], v0.4h, #8
; CHECK-COMMON-NEXT:  sshr [[OP2:v[0-9]+\.4h]], [[OP1]], #8
; CHECK-FP16-NEXT:    scvtf v0.4h, [[OP2]]
; CHECK-CVT-NEXT:     sshll [[OP3:v[0-9]+\.4s]], [[OP2]], #0
; CHECK-CVT-NEXT:     scvtf [[OP4:v[0-9]+\.4s]], [[OP3]]
; CHECK-CVT-NEXT:     fcvtn v0.4h, [[OP4]]
; CHECK-COMMON-NEXT:  ret
  %1 = sitofp <4 x i8> %a to <4 x half>
  ret <4 x half> %1
}

define <4 x half> @sitofp_i16(<4 x i16> %a) #0 {
; CHECK-COMMON-LABEL: sitofp_i16:
; CHECK-FP16-NEXT:   scvtf v0.4h, v0.4h
; CHECK-CVT-NEXT:    sshll [[OP1:v[0-9]+\.4s]], v0.4h, #0
; CHECK-CVT-NEXT:    scvtf [[OP2:v[0-9]+\.4s]], [[OP1]]
; CHECK-CVT-NEXT:    fcvtn v0.4h, [[OP2]]
; CHECK-COMMON-NEXT: ret
  %1 = sitofp <4 x i16> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @sitofp_i32(<4 x i32> %a) #0 {
; CHECK-COMMON-LABEL: sitofp_i32:
; CHECK-COMMON-NEXT:  scvtf [[OP1:v[0-9]+\.4s]], v0.4s
; CHECK-COMMON-NEXT:  fcvtn v0.4h, [[OP1]]
; CHECK-COMMON-NEXT:  ret
  %1 = sitofp <4 x i32> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @sitofp_i64(<4 x i64> %a) #0 {
; CHECK-COMMON-LABEL: sitofp_i64:
; CHECK-COMMON-DAG:   scvtf [[OP1:v[0-9]+\.2d]], v0.2d
; CHECK-COMMON-DAG:   scvtf [[OP2:v[0-9]+\.2d]], v1.2d
; CHECK-COMMON-DAG:   fcvtn [[OP3:v[0-9]+]].2s, [[OP1]]
; CHECK-COMMON-NEXT:  fcvtn2 [[OP3]].4s, [[OP2]]
; CHECK-COMMON-NEXT:  fcvtn v0.4h, [[OP3]].4s
; CHECK-COMMON-NEXT:  ret
  %1 = sitofp <4 x i64> %a to <4 x half>
  ret <4 x half> %1
}

define <4 x half> @uitofp_i8(<4 x i8> %a) #0 {
; CHECK-COMMON-LABEL: uitofp_i8:
; CHECK-COMMON-NEXT:  bic v0.4h, #255, lsl #8
; CHECK-FP16-NEXT:    ucvtf v0.4h, v0.4h
; CHECK-CVT-NEXT:     ushll [[OP1:v[0-9]+\.4s]], v0.4h, #0
; CHECK-CVT-NEXT:     ucvtf [[OP2:v[0-9]+\.4s]], [[OP1]]
; CHECK-CVT-NEXT:     fcvtn v0.4h, [[OP2]]
; CHECK-COMMON-NEXT:  ret
  %1 = uitofp <4 x i8> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @uitofp_i16(<4 x i16> %a) #0 {
; CHECK-COMMON-LABEL: uitofp_i16:
; CHECK-FP16-NEXT:  ucvtf v0.4h, v0.4h
; CHECK-CVT-NEXT:   ushll [[OP1:v[0-9]+\.4s]], v0.4h, #0
; CHECK-CVT-NEXT:   ucvtf [[OP2:v[0-9]+\.4s]], [[OP1]]
; CHECK-CVT-NEXT:   fcvtn v0.4h, [[OP2]]
; CHECK-COMMON-NEXT:  ret
  %1 = uitofp <4 x i16> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @uitofp_i32(<4 x i32> %a) #0 {
; CHECK-COMMON-LABEL: uitofp_i32:
; CHECK-COMMON-NEXT:  ucvtf [[OP1:v[0-9]+\.4s]], v0.4s
; CHECK-COMMON-NEXT:  fcvtn v0.4h, [[OP1]]
; CHECK-COMMON-NEXT:  ret
  %1 = uitofp <4 x i32> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @uitofp_i64(<4 x i64> %a) #0 {
; CHECK-COMMON-LABEL: uitofp_i64:
; CHECK-COMMON-DAG:   ucvtf [[OP1:v[0-9]+\.2d]], v0.2d
; CHECK-COMMON-DAG:   ucvtf [[OP2:v[0-9]+\.2d]], v1.2d
; CHECK-COMMON-DAG:   fcvtn [[OP3:v[0-9]+]].2s, [[OP1]]
; CHECK-COMMON-NEXT:  fcvtn2 [[OP3]].4s, [[OP2]]
; CHECK-COMMON-NEXT:  fcvtn v0.4h, [[OP3]].4s
; CHECK-COMMON-NEXT:  ret
  %1 = uitofp <4 x i64> %a to <4 x half>
  ret <4 x half> %1
}

define void @test_insert_at_zero(half %a, <4 x half>* %b) #0 {
; CHECK-COMMON-LABEL: test_insert_at_zero:
; CHECK-COMMON-NEXT:  str d0, [x0]
; CHECK-COMMON-NEXT:  ret
  %1 = insertelement <4 x half> undef, half %a, i64 0
  store <4 x half> %1, <4 x half>* %b, align 4
  ret void
}

define <4 x i8> @fptosi_i8(<4 x half> %a) #0 {
; CHECK-COMMON-LABEL: fptosi_i8:
; CHECK-COMMON-NEXT:  fcvtl  [[REG1:v[0-9]+\.4s]], v0.4h
; CHECK-COMMON-NEXT:  fcvtzs [[REG2:v[0-9]+\.4s]], [[REG1]]
; CHECK-COMMON-NEXT:  xtn    v0.4h, [[REG2]]
; CHECK-COMMON-NEXT:  ret
  %1 = fptosi<4 x half> %a to <4 x i8>
  ret <4 x i8> %1
}

define <4 x i16> @fptosi_i16(<4 x half> %a) #0 {
; CHECK-COMMON-LABEL: fptosi_i16:
; CHECK-COMMON-NEXT:  fcvtl  [[REG1:v[0-9]+\.4s]], v0.4h
; CHECK-COMMON-NEXT:  fcvtzs [[REG2:v[0-9]+\.4s]], [[REG1]]
; CHECK-COMMON-NEXT:  xtn    v0.4h, [[REG2]]
; CHECK-COMMON-NEXT:  ret
  %1 = fptosi<4 x half> %a to <4 x i16>
  ret <4 x i16> %1
}

define <4 x i8> @fptoui_i8(<4 x half> %a) #0 {
; CHECK-COMMON-LABEL: fptoui_i8:
; CHECK-COMMON-NEXT:  fcvtl  [[REG1:v[0-9]+\.4s]], v0.4h
; NOTE: fcvtzs selected here because the xtn shaves the sign bit
; CHECK-COMMON-NEXT:  fcvtzs [[REG2:v[0-9]+\.4s]], [[REG1]]
; CHECK-COMMON-NEXT:  xtn    v0.4h, [[REG2]]
; CHECK-COMMON-NEXT:  ret
  %1 = fptoui<4 x half> %a to <4 x i8>
  ret <4 x i8> %1
}

define <4 x i16> @fptoui_i16(<4 x half> %a) #0 {
; CHECK-COMMON-LABEL: fptoui_i16:
; CHECK-COMMON-NEXT:  fcvtl  [[REG1:v[0-9]+\.4s]], v0.4h
; CHECK-COMMON-NEXT:  fcvtzu [[REG2:v[0-9]+\.4s]], [[REG1]]
; CHECK-COMMON-NEXT:  xtn    v0.4h, [[REG2]]
; CHECK-COMMON-NEXT:  ret
  %1 = fptoui<4 x half> %a to <4 x i16>
  ret <4 x i16> %1
}

define <4 x i1> @test_fcmp_une(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_une:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmeq
; CHECK-CVT: mvn
; CHECK-CVT: xtn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_une:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmeq v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp une <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_ueq(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_ueq:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmgt
; CHECK-CVT: fcmgt
; CHECK-CVT: orr
; CHECK-CVT: xtn
; CHECK-CVT: mvn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_ueq:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp ueq <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_ugt(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_ugt:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmge
; CHECK-CVT: xtn
; CHECK-CVT: mvn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_ugt:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmge v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp ugt <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_uge(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_uge:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmgt
; CHECK-CVT: xtn
; CHECK-CVT: mvn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_uge:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp uge <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_ult(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_ult:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmge
; CHECK-CVT: xtn
; CHECK-CVT: mvn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_ult:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmge v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp ult <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_ule(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_ule:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmgt
; CHECK-CVT: xtn
; CHECK-CVT: mvn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_ule:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp ule <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_uno(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_uno:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmge
; CHECK-CVT: fcmgt
; CHECK-CVT: orr
; CHECK-CVT: xtn
; CHECK-CVT: mvn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_uno:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmge v{{[0-9]}}.4h, v{{[0-9]}}.4h
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp uno <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_one(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_one:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmgt
; CHECK-CVT: fcmgt
; CHECK-CVT: orr
; CHECK-CVT: xtn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_one:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp one <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_oeq(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_oeq:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmeq
; CHECK-CVT: xtn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_oeq:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmeq v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp oeq <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_ogt(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_ogt:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmgt
; CHECK-CVT: xtn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_ogt:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp ogt <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_oge(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_oge:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmge
; CHECK-CVT: xtn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_oge:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmge v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp oge <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_olt(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_olt:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmgt
; CHECK-CVT: xtn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_olt:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp olt <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_ole(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_ole:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmge
; CHECK-CVT: xtn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_ole:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmge v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp ole <4 x half> %a, %b
  ret <4 x i1> %1
}

define <4 x i1> @test_fcmp_ord(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-CVT-LABEL: test_fcmp_ord:
; CHECK-CVT: fcvtl
; CHECK-CVT: fcvtl
; CHECK-CVT: fcmge
; CHECK-CVT: fcmgt
; CHECK-CVT: orr
; CHECK-CVT: xtn
; CHECK-CVT: ret

; CHECK-FP16-LABEL: test_fcmp_ord:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16:       fcmge v{{[0-9]}}.4h, v{{[0-9]}}.4h
; CHECK-FP16:       fcmgt v{{[0-9]}}.4h, v{{[0-9]}}.4h
  %1 = fcmp ord <4 x half> %a, %b
  ret <4 x i1> %1
}

attributes #0 = { nounwind }
