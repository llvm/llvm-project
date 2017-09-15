; RUN: llc < %s -asm-verbose=false -mtriple=aarch64-none-eabi -mattr=-fullfp16 | FileCheck %s --check-prefix=CHECK-CVT   --check-prefix=CHECK
; RUN: llc < %s -asm-verbose=false -mtriple=aarch64-none-eabi -mattr=+fullfp16 | FileCheck %s --check-prefix=CHECK-FP16  --check-prefix=CHECK

define <8 x half> @add_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-CVT-LABEL: add_h:
; CHECK-CVT:     fcvt
; CHECK-CVT:     fcvt
; CHECK-CVT-DAG: fadd
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fadd
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fadd
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fadd
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fadd
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fadd
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fadd
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fadd
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT-DAG: fcvt
; CHECK-CVT:     fcvt

; CHECK-FP16-LABEL: add_h:
; CHECK-FP16:       fadd  v0.8h, v0.8h, v1.8h
; CHECK-FP16-NEXT:  ret

  %0 = fadd <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @sub_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-CVT-LABEL: sub_h:
; CHECK-CVT:       fcvt
; CHECK-CVT:       fcvt
; CHECK-CVT-DAG:   fsub
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fsub
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fsub
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fsub
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fsub
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fsub
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fsub
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fsub
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT:       fcvt

; CHECK-FP16-LABEL: sub_h:
; CHECK-FP16:       fsub  v0.8h, v0.8h, v1.8h
; CHECK-FP16-NEXT:  ret

  %0 = fsub <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @mul_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-CVT-LABEL: mul_h:
; CHECK-CVT:       fcvt
; CHECK-CVT:       fcvt
; CHECK-CVT-DAG:   fmul
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fmul
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fmul
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fmul
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fmul
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fmul
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fmul
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fmul
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT:       fcvt

; CHECK-FP16-LABEL: mul_h:
; CHECK-FP16:       fmul  v0.8h, v0.8h, v1.8h
; CHECK-FP16-NEXT:  ret

  %0 = fmul <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @div_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-CVT-LABEL: div_h:
; CHECK-CVT:       fcvt
; CHECK-CVT:       fcvt
; CHECK-CVT-DAG:   fdiv
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fdiv
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fdiv
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fdiv
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fdiv
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fdiv
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fdiv
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fdiv
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT-DAG:   fcvt
; CHECK-CVT:       fcvt

; CHECK-FP16-LABEL: div_h:
; CHECK-FP16:       fdiv  v0.8h, v0.8h, v1.8h
; CHECK-FP16-NEXT:  ret

  %0 = fdiv <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @load_h(<8 x half>* %a) {
entry:
; CHECK-LABEL: load_h:
; CHECK: ldr q0, [x0]
  %0 = load <8 x half>, <8 x half>* %a, align 4
  ret <8 x half> %0
}


define void @store_h(<8 x half>* %a, <8 x half> %b) {
entry:
; CHECK-LABEL: store_h:
; CHECK: str q0, [x0]
  store <8 x half> %b, <8 x half>* %a, align 4
  ret void
}

define <8 x half> @s_to_h(<8 x float> %a) {
; CHECK-LABEL: s_to_h:
; CHECK-DAG: fcvtn v0.4h, v0.4s
; CHECK-DAG: fcvtn [[REG:v[0-9+]]].4h, v1.4s
; CHECK: ins v0.d[1], [[REG]].d[0]
  %1 = fptrunc <8 x float> %a to <8 x half>
  ret <8 x half> %1
}

define <8 x half> @d_to_h(<8 x double> %a) {
; CHECK-LABEL: d_to_h:
; CHECK-DAG: mov d{{[0-9]+}}, v{{[0-9]+}}.d[1]
; CHECK-DAG: mov d{{[0-9]+}}, v{{[0-9]+}}.d[1]
; CHECK-DAG: mov d{{[0-9]+}}, v{{[0-9]+}}.d[1]
; CHECK-DAG: mov d{{[0-9]+}}, v{{[0-9]+}}.d[1]
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
  %1 = fptrunc <8 x double> %a to <8 x half>
  ret <8 x half> %1
}

define <8 x float> @h_to_s(<8 x half> %a) {
; CHECK-LABEL: h_to_s:
; CHECK: fcvtl2 v1.4s, v0.8h
; CHECK: fcvtl v0.4s, v0.4h
  %1 = fpext <8 x half> %a to <8 x float>
  ret <8 x float> %1
}

define <8 x double> @h_to_d(<8 x half> %a) {
; CHECK-LABEL: h_to_d:
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
  %1 = fpext <8 x half> %a to <8 x double>
  ret <8 x double> %1
}


define <8 x half> @bitcast_i_to_h(float, <8 x i16> %a) {
; CHECK-LABEL: bitcast_i_to_h:
; CHECK: mov v0.16b, v1.16b
  %2 = bitcast <8 x i16> %a to <8 x half>
  ret <8 x half> %2
}

define <8 x i16> @bitcast_h_to_i(float, <8 x half> %a) {
; CHECK-LABEL: bitcast_h_to_i:
; CHECK: mov v0.16b, v1.16b
  %2 = bitcast <8 x half> %a to <8 x i16>
  ret <8 x i16> %2
}


define <8 x half> @sitofp_i8(<8 x i8> %a) #0 {
; CHECK-LABEL: sitofp_i8:
; CHECK-NEXT: sshll v[[REG1:[0-9]+]].8h, v0.8b, #0
; CHECK-NEXT: sshll2 [[LO:v[0-9]+\.4s]], v[[REG1]].8h, #0
; CHECK-NEXT: sshll  [[HI:v[0-9]+\.4s]], v[[REG1]].4h, #0
; CHECK-DAG: scvtf [[HIF:v[0-9]+\.4s]], [[HI]]
; CHECK-DAG: scvtf [[LOF:v[0-9]+\.4s]], [[LO]]
; CHECK-DAG: fcvtn v[[LOREG:[0-9]+]].4h, [[LOF]]
; CHECK-DAG: fcvtn v0.4h, [[HIF]]
; CHECK: ins v0.d[1], v[[LOREG]].d[0]
  %1 = sitofp <8 x i8> %a to <8 x half>
  ret <8 x half> %1
}


define <8 x half> @sitofp_i16(<8 x i16> %a) #0 {
; CHECK-LABEL: sitofp_i16:
; CHECK-NEXT: sshll2 [[LO:v[0-9]+\.4s]], v0.8h, #0
; CHECK-NEXT: sshll  [[HI:v[0-9]+\.4s]], v0.4h, #0
; CHECK-DAG: scvtf [[HIF:v[0-9]+\.4s]], [[HI]]
; CHECK-DAG: scvtf [[LOF:v[0-9]+\.4s]], [[LO]]
; CHECK-DAG: fcvtn v[[LOREG:[0-9]+]].4h, [[LOF]]
; CHECK-DAG: fcvtn v0.4h, [[HIF]]
; CHECK: ins v0.d[1], v[[LOREG]].d[0]
  %1 = sitofp <8 x i16> %a to <8 x half>
  ret <8 x half> %1
}


define <8 x half> @sitofp_i32(<8 x i32> %a) #0 {
; CHECK-LABEL: sitofp_i32:
; CHECK-DAG: scvtf [[OP1:v[0-9]+\.4s]], v0.4s
; CHECK-DAG: scvtf [[OP2:v[0-9]+\.4s]], v1.4s
; CHECK-DAG: fcvtn v[[REG:[0-9]+]].4h, [[OP2]]
; CHECK-DAG: fcvtn v0.4h, [[OP1]]
; CHECK: ins v0.d[1], v[[REG]].d[0]
  %1 = sitofp <8 x i32> %a to <8 x half>
  ret <8 x half> %1
}


define <8 x half> @sitofp_i64(<8 x i64> %a) #0 {
; CHECK-LABEL: sitofp_i64:
; CHECK-DAG: scvtf [[OP1:v[0-9]+\.2d]], v0.2d
; CHECK-DAG: scvtf [[OP2:v[0-9]+\.2d]], v1.2d
; CHECK-DAG: fcvtn [[OP3:v[0-9]+]].2s, [[OP1]]
; CHECK-DAG: fcvtn2 [[OP3]].4s, [[OP2]]
; CHECK: fcvtn v0.4h, [[OP3]].4s
  %1 = sitofp <8 x i64> %a to <8 x half>
  ret <8 x half> %1
}

define <8 x half> @uitofp_i8(<8 x i8> %a) #0 {
; CHECK-LABEL: uitofp_i8:
; CHECK-NEXT: ushll v[[REG1:[0-9]+]].8h, v0.8b, #0
; CHECK-NEXT: ushll2 [[LO:v[0-9]+\.4s]], v[[REG1]].8h, #0
; CHECK-NEXT: ushll  [[HI:v[0-9]+\.4s]], v[[REG1]].4h, #0
; CHECK-DAG: ucvtf [[HIF:v[0-9]+\.4s]], [[HI]]
; CHECK-DAG: ucvtf [[LOF:v[0-9]+\.4s]], [[LO]]
; CHECK-DAG: fcvtn v[[LOREG:[0-9]+]].4h, [[LOF]]
; CHECK-DAG: fcvtn v0.4h, [[HIF]]
; CHECK: ins v0.d[1], v[[LOREG]].d[0]
  %1 = uitofp <8 x i8> %a to <8 x half>
  ret <8 x half> %1
}


define <8 x half> @uitofp_i16(<8 x i16> %a) #0 {
; CHECK-LABEL: uitofp_i16:
; CHECK-NEXT: ushll2 [[LO:v[0-9]+\.4s]], v0.8h, #0
; CHECK-NEXT: ushll  [[HI:v[0-9]+\.4s]], v0.4h, #0
; CHECK-DAG: ucvtf [[HIF:v[0-9]+\.4s]], [[HI]]
; CHECK-DAG: ucvtf [[LOF:v[0-9]+\.4s]], [[LO]]
; CHECK-DAG: fcvtn v[[LOREG:[0-9]+]].4h, [[LOF]]
; CHECK-DAG: fcvtn v0.4h, [[HIF]]
; CHECK: ins v0.d[1], v[[LOREG]].d[0]
  %1 = uitofp <8 x i16> %a to <8 x half>
  ret <8 x half> %1
}


define <8 x half> @uitofp_i32(<8 x i32> %a) #0 {
; CHECK-LABEL: uitofp_i32:
; CHECK-DAG: ucvtf [[OP1:v[0-9]+\.4s]], v0.4s
; CHECK-DAG: ucvtf [[OP2:v[0-9]+\.4s]], v1.4s
; CHECK-DAG: fcvtn v[[REG:[0-9]+]].4h, [[OP2]]
; CHECK-DAG: fcvtn v0.4h, [[OP1]]
; CHECK: ins v0.d[1], v[[REG]].d[0]
  %1 = uitofp <8 x i32> %a to <8 x half>
  ret <8 x half> %1
}


define <8 x half> @uitofp_i64(<8 x i64> %a) #0 {
; CHECK-LABEL: uitofp_i64:
; CHECK-DAG: ucvtf [[OP1:v[0-9]+\.2d]], v0.2d
; CHECK-DAG: ucvtf [[OP2:v[0-9]+\.2d]], v1.2d
; CHECK-DAG: fcvtn [[OP3:v[0-9]+]].2s, [[OP1]]
; CHECK-DAG: fcvtn2 [[OP3]].4s, [[OP2]]
; CHECK: fcvtn v0.4h, [[OP3]].4s
  %1 = uitofp <8 x i64> %a to <8 x half>
  ret <8 x half> %1
}

define void @test_insert_at_zero(half %a, <8 x half>* %b) #0 {
; CHECK-LABEL: test_insert_at_zero:
; CHECK-NEXT: str q0, [x0]
; CHECK-NEXT: ret
  %1 = insertelement <8 x half> undef, half %a, i64 0
  store <8 x half> %1, <8 x half>* %b, align 4
  ret void
}

define <8 x i8> @fptosi_i8(<8 x half> %a) #0 {
; CHECK-LABEL: fptosi_i8:
; CHECK-DAG: fcvtl   [[LO:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl2  [[HI:v[0-9]+\.4s]], v0.8h
; CHECK-DAG: fcvtzs  [[LOF32:v[0-9]+\.4s]], [[LO]]
; CHECK-DAG: xtn     [[I16:v[0-9]+]].4h, [[LOF32]]
; CHECK-DAG: fcvtzs  [[HIF32:v[0-9]+\.4s]], [[HI]]
; CHECK-DAG: xtn2    [[I16]].8h, [[HIF32]]
; CHECK-NEXT: xtn     v0.8b, [[I16]].8h
; CHECK-NEXT: ret
  %1 = fptosi<8 x half> %a to <8 x i8>
  ret <8 x i8> %1
}

define <8 x i16> @fptosi_i16(<8 x half> %a) #0 {
; CHECK-LABEL: fptosi_i16:
; CHECK-DAG: fcvtl   [[LO:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl2  [[HI:v[0-9]+\.4s]], v0.8h
; CHECK-DAG: fcvtzs  [[LOF32:v[0-9]+\.4s]], [[LO]]
; CHECK-DAG: xtn     [[I16:v[0-9]+]].4h, [[LOF32]]
; CHECK-DAG: fcvtzs  [[HIF32:v[0-9]+\.4s]], [[HI]]
; CHECK-NEXT: xtn2    [[I16]].8h, [[HIF32]]
; CHECK-NEXT: ret
  %1 = fptosi<8 x half> %a to <8 x i16>
  ret <8 x i16> %1
}

define <8 x i8> @fptoui_i8(<8 x half> %a) #0 {
; CHECK-LABEL: fptoui_i8:
; CHECK-DAG: fcvtl   [[LO:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl2  [[HI:v[0-9]+\.4s]], v0.8h
; CHECK-DAG: fcvtzu  [[LOF32:v[0-9]+\.4s]], [[LO]]
; CHECK-DAG: xtn     [[I16:v[0-9]+]].4h, [[LOF32]]
; CHECK-DAG: fcvtzu  [[HIF32:v[0-9]+\.4s]], [[HI]]
; CHECK-DAG: xtn2    [[I16]].8h, [[HIF32]]
; CHECK-NEXT: xtn     v0.8b, [[I16]].8h
; CHECK-NEXT: ret
  %1 = fptoui<8 x half> %a to <8 x i8>
  ret <8 x i8> %1
}

define <8 x i16> @fptoui_i16(<8 x half> %a) #0 {
; CHECK-LABEL: fptoui_i16:
; CHECK-DAG: fcvtl   [[LO:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl2  [[HI:v[0-9]+\.4s]], v0.8h
; CHECK-DAG: fcvtzu  [[LOF32:v[0-9]+\.4s]], [[LO]]
; CHECK-DAG: xtn     [[I16:v[0-9]+]].4h, [[LOF32]]
; CHECK-DAG: fcvtzu  [[HIF32:v[0-9]+\.4s]], [[HI]]
; CHECK-NEXT: xtn2    [[I16]].8h, [[HIF32]]
; CHECK-NEXT: ret
  %1 = fptoui<8 x half> %a to <8 x i16>
  ret <8 x i16> %1
}

define <8 x i1> @test_fcmp_une(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_une:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp une <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_ueq(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_ueq:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp ueq <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_ugt(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_ugt:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp ugt <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_uge(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_uge:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp uge <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_ult(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_ult:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp ult <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_ule(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_ule:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp ule <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_uno(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_uno:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp uno <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_one(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_one:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp one <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_oeq(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_oeq:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp oeq <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_ogt(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_ogt:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp ogt <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_oge(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_oge:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp oge <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_olt(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_olt:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp olt <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_ole(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_ole:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp ole <8 x half> %a, %b
  ret <8 x i1> %1
}

define <8 x i1> @test_fcmp_ord(<8 x half> %a, <8 x half> %b) #0 {
; FileCheck checks are unwieldy with 16 fcvt and 8 csel tests, so skipped for -fullfp16.

; CHECK-FP16-LABEL: test_fcmp_ord:
; CHECK-FP16-NOT:   fcvt
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}
; CHECK-FP16-DAG:   fcmp  h{{[0-9]}}, h{{[0-9]}}

  %1 = fcmp ord <8 x half> %a, %b
  ret <8 x i1> %1
}

attributes #0 = { nounwind }
