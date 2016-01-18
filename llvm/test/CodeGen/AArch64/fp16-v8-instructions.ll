; RUN: llc < %s -asm-verbose=false -mtriple=aarch64-none-eabi | FileCheck %s

define <8 x half> @add_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-LABEL: add_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK: fcvt
  %0 = fadd <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @sub_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-LABEL: sub_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK: fcvt
  %0 = fsub <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @mul_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-LABEL: mul_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK: fcvt
  %0 = fmul <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @div_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-LABEL: div_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK: fcvt
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

attributes #0 = { nounwind }
