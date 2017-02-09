; RUN: llc < %s -asm-verbose=false -mtriple=aarch64-none-eabi | FileCheck %s

define <4 x half> @add_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-LABEL: add_h:
; CHECK-DAG: fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK: fadd [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK: fcvtn v0.4h, [[RES]]
  %0 = fadd <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @build_h4(<4 x half> %a) {
entry:
; CHECK-LABEL: build_h4:
; CHECK: mov [[GPR:w[0-9]+]], #15565
; CHECK: dup v0.4h, [[GPR]]
  ret <4 x half> <half 0xH3CCD, half 0xH3CCD, half 0xH3CCD, half 0xH3CCD>
}


define <4 x half> @sub_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-LABEL: sub_h:
; CHECK-DAG: fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK: fsub [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK: fcvtn v0.4h, [[RES]]
  %0 = fsub <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @mul_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-LABEL: mul_h:
; CHECK-DAG: fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK: fmul [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK: fcvtn v0.4h, [[RES]]
  %0 = fmul <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @div_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-LABEL: div_h:
; CHECK-DAG: fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK: fdiv [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK: fcvtn v0.4h, [[RES]]
  %0 = fdiv <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @load_h(<4 x half>* %a) {
entry:
; CHECK-LABEL: load_h:
; CHECK: ldr d0, [x0]
  %0 = load <4 x half>, <4 x half>* %a, align 4
  ret <4 x half> %0
}


define void @store_h(<4 x half>* %a, <4 x half> %b) {
entry:
; CHECK-LABEL: store_h:
; CHECK: str d0, [x0]
  store <4 x half> %b, <4 x half>* %a, align 4
  ret void
}

define <4 x half> @s_to_h(<4 x float> %a) {
; CHECK-LABEL: s_to_h:
; CHECK: fcvtn v0.4h, v0.4s
  %1 = fptrunc <4 x float> %a to <4 x half>
  ret <4 x half> %1
}

define <4 x half> @d_to_h(<4 x double> %a) {
; CHECK-LABEL: d_to_h:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
  %1 = fptrunc <4 x double> %a to <4 x half>
  ret <4 x half> %1
}

define <4 x float> @h_to_s(<4 x half> %a) {
; CHECK-LABEL: h_to_s:
; CHECK: fcvtl v0.4s, v0.4h
  %1 = fpext <4 x half> %a to <4 x float>
  ret <4 x float> %1
}

define <4 x double> @h_to_d(<4 x half> %a) {
; CHECK-LABEL: h_to_d:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
  %1 = fpext <4 x half> %a to <4 x double>
  ret <4 x double> %1
}

define <4 x half> @bitcast_i_to_h(float, <4 x i16> %a) {
; CHECK-LABEL: bitcast_i_to_h:
; CHECK: mov v0.16b, v1.16b
  %2 = bitcast <4 x i16> %a to <4 x half>
  ret <4 x half> %2
}

define <4 x i16> @bitcast_h_to_i(float, <4 x half> %a) {
; CHECK-LABEL: bitcast_h_to_i:
; CHECK: mov v0.16b, v1.16b
  %2 = bitcast <4 x half> %a to <4 x i16>
  ret <4 x i16> %2
}

define <4 x half> @sitofp_i8(<4 x i8> %a) #0 {
; CHECK-LABEL: sitofp_i8:
; CHECK-NEXT: shl [[OP1:v[0-9]+\.4h]], v0.4h, #8
; CHECK-NEXT: sshr [[OP2:v[0-9]+\.4h]], [[OP1]], #8
; CHECK-NEXT: sshll [[OP3:v[0-9]+\.4s]], [[OP2]], #0
; CHECK-NEXT: scvtf [[OP4:v[0-9]+\.4s]], [[OP3]]
; CHECK-NEXT: fcvtn v0.4h, [[OP4]]
; CHECK-NEXT: ret
  %1 = sitofp <4 x i8> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @sitofp_i16(<4 x i16> %a) #0 {
; CHECK-LABEL: sitofp_i16:
; CHECK-NEXT: sshll [[OP1:v[0-9]+\.4s]], v0.4h, #0
; CHECK-NEXT: scvtf [[OP2:v[0-9]+\.4s]], [[OP1]]
; CHECK-NEXT: fcvtn v0.4h, [[OP2]]
; CHECK-NEXT: ret
  %1 = sitofp <4 x i16> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @sitofp_i32(<4 x i32> %a) #0 {
; CHECK-LABEL: sitofp_i32:
; CHECK-NEXT: scvtf [[OP1:v[0-9]+\.4s]], v0.4s
; CHECK-NEXT: fcvtn v0.4h, [[OP1]]
  %1 = sitofp <4 x i32> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @sitofp_i64(<4 x i64> %a) #0 {
; CHECK-LABEL: sitofp_i64:
; CHECK-DAG: scvtf [[OP1:v[0-9]+\.2d]], v0.2d
; CHECK-DAG: scvtf [[OP2:v[0-9]+\.2d]], v1.2d
; CHECK-DAG: fcvtn [[OP3:v[0-9]+]].2s, [[OP1]]
; CHECK-NEXT: fcvtn2 [[OP3]].4s, [[OP2]]
; CHECK-NEXT: fcvtn v0.4h, [[OP3]].4s
  %1 = sitofp <4 x i64> %a to <4 x half>
  ret <4 x half> %1
}

define <4 x half> @uitofp_i8(<4 x i8> %a) #0 {
; CHECK-LABEL: uitofp_i8:
; CHECK-NEXT: bic v0.4h, #255, lsl #8
; CHECK-NEXT: ushll [[OP1:v[0-9]+\.4s]], v0.4h, #0
; CHECK-NEXT: ucvtf [[OP2:v[0-9]+\.4s]], [[OP1]]
; CHECK-NEXT: fcvtn v0.4h, [[OP2]]
; CHECK-NEXT: ret
  %1 = uitofp <4 x i8> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @uitofp_i16(<4 x i16> %a) #0 {
; CHECK-LABEL: uitofp_i16:
; CHECK-NEXT: ushll [[OP1:v[0-9]+\.4s]], v0.4h, #0
; CHECK-NEXT: ucvtf [[OP2:v[0-9]+\.4s]], [[OP1]]
; CHECK-NEXT: fcvtn v0.4h, [[OP2]]
; CHECK-NEXT: ret
  %1 = uitofp <4 x i16> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @uitofp_i32(<4 x i32> %a) #0 {
; CHECK-LABEL: uitofp_i32:
; CHECK-NEXT: ucvtf [[OP1:v[0-9]+\.4s]], v0.4s
; CHECK-NEXT: fcvtn v0.4h, [[OP1]]
  %1 = uitofp <4 x i32> %a to <4 x half>
  ret <4 x half> %1
}


define <4 x half> @uitofp_i64(<4 x i64> %a) #0 {
; CHECK-LABEL: uitofp_i64:
; CHECK-DAG: ucvtf [[OP1:v[0-9]+\.2d]], v0.2d
; CHECK-DAG: ucvtf [[OP2:v[0-9]+\.2d]], v1.2d
; CHECK-DAG: fcvtn [[OP3:v[0-9]+]].2s, [[OP1]]
; CHECK-NEXT: fcvtn2 [[OP3]].4s, [[OP2]]
; CHECK-NEXT: fcvtn v0.4h, [[OP3]].4s
  %1 = uitofp <4 x i64> %a to <4 x half>
  ret <4 x half> %1
}

define void @test_insert_at_zero(half %a, <4 x half>* %b) #0 {
; CHECK-LABEL: test_insert_at_zero:
; CHECK-NEXT: str d0, [x0]
; CHECK-NEXT: ret
  %1 = insertelement <4 x half> undef, half %a, i64 0
  store <4 x half> %1, <4 x half>* %b, align 4
  ret void
}

define <4 x i8> @fptosi_i8(<4 x half> %a) #0 {
; CHECK-LABEL: fptosi_i8:
; CHECK-NEXT: fcvtl  [[REG1:v[0-9]+\.4s]], v0.4h
; CHECK-NEXT: fcvtzs [[REG2:v[0-9]+\.4s]], [[REG1]]
; CHECK-NEXT: xtn    v0.4h, [[REG2]]
; CHECK-NEXT: ret
  %1 = fptosi<4 x half> %a to <4 x i8>
  ret <4 x i8> %1
}

define <4 x i16> @fptosi_i16(<4 x half> %a) #0 {
; CHECK-LABEL: fptosi_i16:
; CHECK-NEXT: fcvtl  [[REG1:v[0-9]+\.4s]], v0.4h
; CHECK-NEXT: fcvtzs [[REG2:v[0-9]+\.4s]], [[REG1]]
; CHECK-NEXT: xtn    v0.4h, [[REG2]]
; CHECK-NEXT: ret
  %1 = fptosi<4 x half> %a to <4 x i16>
  ret <4 x i16> %1
}

define <4 x i8> @fptoui_i8(<4 x half> %a) #0 {
; CHECK-LABEL: fptoui_i8:
; CHECK-NEXT: fcvtl  [[REG1:v[0-9]+\.4s]], v0.4h
; NOTE: fcvtzs selected here because the xtn shaves the sign bit
; CHECK-NEXT: fcvtzs [[REG2:v[0-9]+\.4s]], [[REG1]]
; CHECK-NEXT: xtn    v0.4h, [[REG2]]
; CHECK-NEXT: ret
  %1 = fptoui<4 x half> %a to <4 x i8>
  ret <4 x i8> %1
}

define <4 x i16> @fptoui_i16(<4 x half> %a) #0 {
; CHECK-LABEL: fptoui_i16:
; CHECK-NEXT: fcvtl  [[REG1:v[0-9]+\.4s]], v0.4h
; CHECK-NEXT: fcvtzu [[REG2:v[0-9]+\.4s]], [[REG1]]
; CHECK-NEXT: xtn    v0.4h, [[REG2]]
; CHECK-NEXT: ret
  %1 = fptoui<4 x half> %a to <4 x i16>
  ret <4 x i16> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_une:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, ne
; CHECK-DAG: csetm {{.*}}, ne
; CHECK-DAG: csetm {{.*}}, ne
; CHECK-DAG: csetm {{.*}}, ne
define <4 x i1> @test_fcmp_une(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp une <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_ueq:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm [[REG1:w[0-9]+]], eq
; CHECK-DAG: csetm [[REG2:w[0-9]+]], eq
; CHECK-DAG: csetm [[REG3:w[0-9]+]], eq
; CHECK-DAG: csetm [[REG4:w[0-9]+]], eq
; CHECK-DAG: csinv {{.*}}, [[REG1]], wzr, vc
; CHECK-DAG: csinv {{.*}}, [[REG2]], wzr, vc
; CHECK-DAG: csinv {{.*}}, [[REG3]], wzr, vc
; CHECK-DAG: csinv {{.*}}, [[REG4]], wzr, vc
define <4 x i1> @test_fcmp_ueq(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp ueq <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_ugt:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, hi
; CHECK-DAG: csetm {{.*}}, hi
; CHECK-DAG: csetm {{.*}}, hi
; CHECK-DAG: csetm {{.*}}, hi
define <4 x i1> @test_fcmp_ugt(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp ugt <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_uge:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, pl
; CHECK-DAG: csetm {{.*}}, pl
; CHECK-DAG: csetm {{.*}}, pl
; CHECK-DAG: csetm {{.*}}, pl
define <4 x i1> @test_fcmp_uge(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp uge <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_ult:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, lt
; CHECK-DAG: csetm {{.*}}, lt
; CHECK-DAG: csetm {{.*}}, lt
; CHECK-DAG: csetm {{.*}}, lt
define <4 x i1> @test_fcmp_ult(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp ult <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_ule:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, le
; CHECK-DAG: csetm {{.*}}, le
; CHECK-DAG: csetm {{.*}}, le
; CHECK-DAG: csetm {{.*}}, le
define <4 x i1> @test_fcmp_ule(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp ule <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_uno:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, vs
; CHECK-DAG: csetm {{.*}}, vs
; CHECK-DAG: csetm {{.*}}, vs
; CHECK-DAG: csetm {{.*}}, vs
define <4 x i1> @test_fcmp_uno(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp uno <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_one:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm [[REG1:w[0-9]+]], mi
; CHECK-DAG: csetm [[REG2:w[0-9]+]], mi
; CHECK-DAG: csetm [[REG3:w[0-9]+]], mi
; CHECK-DAG: csetm [[REG4:w[0-9]+]], mi
; CHECK-DAG: csinv {{.*}}, [[REG1]], wzr, le
; CHECK-DAG: csinv {{.*}}, [[REG2]], wzr, le
; CHECK-DAG: csinv {{.*}}, [[REG3]], wzr, le
; CHECK-DAG: csinv {{.*}}, [[REG4]], wzr, le

define <4 x i1> @test_fcmp_one(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp one <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_oeq:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, eq
; CHECK-DAG: csetm {{.*}}, eq
; CHECK-DAG: csetm {{.*}}, eq
; CHECK-DAG: csetm {{.*}}, eq
define <4 x i1> @test_fcmp_oeq(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp oeq <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_ogt:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, gt
; CHECK-DAG: csetm {{.*}}, gt
; CHECK-DAG: csetm {{.*}}, gt
; CHECK-DAG: csetm {{.*}}, gt
define <4 x i1> @test_fcmp_ogt(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp ogt <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_oge:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, ge
; CHECK-DAG: csetm {{.*}}, ge
; CHECK-DAG: csetm {{.*}}, ge
; CHECK-DAG: csetm {{.*}}, ge
define <4 x i1> @test_fcmp_oge(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp oge <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_olt:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, mi
; CHECK-DAG: csetm {{.*}}, mi
; CHECK-DAG: csetm {{.*}}, mi
; CHECK-DAG: csetm {{.*}}, mi
define <4 x i1> @test_fcmp_olt(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp olt <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_ole:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, ls
; CHECK-DAG: csetm {{.*}}, ls
; CHECK-DAG: csetm {{.*}}, ls
; CHECK-DAG: csetm {{.*}}, ls
define <4 x i1> @test_fcmp_ole(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp ole <4 x half> %a, %b
  ret <4 x i1> %1
}

; Function Attrs: nounwind readnone
; CHECK-LABEL: test_fcmp_ord:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: csetm {{.*}}, vc
; CHECK-DAG: csetm {{.*}}, vc
; CHECK-DAG: csetm {{.*}}, vc
; CHECK-DAG: csetm {{.*}}, vc
define <4 x i1> @test_fcmp_ord(<4 x half> %a, <4 x half> %b) #0 {
  %1 = fcmp ord <4 x half> %a, %b
  ret <4 x i1> %1
}

attributes #0 = { nounwind }
