; RUN: llc -march=mips -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,O32 %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,O32 %s
; RUN: llc -march=mips64 -target-abi=n32 -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,N32 %s
; RUN: llc -march=mips64el -target-abi=n32 -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,N32 %s
; RUN: llc -march=mips64 -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,N64 %s
; RUN: llc -march=mips64el -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,N64 %s

@v4f32 = global <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>
@v2f64 = global <2 x double> <double 0.0, double 0.0>
@i32 = global i32 0
@f32 = global float 0.0
@f64 = global double 0.0

define void @const_v4f32() nounwind {
  ; ALL-LABEL: const_v4f32:

  store volatile <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>, <4 x float>*@v4f32
  ; ALL: ldi.b  [[R1:\$w[0-9]+]], 0

  store volatile <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, <4 x float>*@v4f32
  ; ALL: lui     [[R1:\$[0-9]+]], 16256
  ; ALL: fill.w  [[R2:\$w[0-9]+]], [[R1]]

  store volatile <4 x float> <float 1.0, float 1.0, float 1.0, float 31.0>, <4 x float>*@v4f32
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; ALL: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <4 x float> <float 65537.0, float 65537.0, float 65537.0, float 65537.0>, <4 x float>*@v4f32
  ; ALL: lui     [[R1:\$[0-9]+]], 18304
  ; ALL: ori     [[R2:\$[0-9]+]], [[R1]], 128
  ; ALL: fill.w  [[R3:\$w[0-9]+]], [[R2]]

  store volatile <4 x float> <float 1.0, float 2.0, float 1.0, float 2.0>, <4 x float>*@v4f32
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; ALL: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <4 x float> <float 3.0, float 4.0, float 5.0, float 6.0>, <4 x float>*@v4f32
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; ALL: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
}

define void @const_v2f64() nounwind {
  ; ALL-LABEL: const_v2f64:

  store volatile <2 x double> <double 0.0, double 0.0>, <2 x double>*@v2f64
  ; ALL: ldi.b  [[R1:\$w[0-9]+]], 0

  store volatile <2 x double> <double 72340172838076673.0, double 72340172838076673.0>, <2 x double>*@v2f64
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; ALL: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 281479271743489.0, double 281479271743489.0>, <2 x double>*@v2f64
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; ALL: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 4294967297.0, double 4294967297.0>, <2 x double>*@v2f64
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; ALL: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 1.0, double 1.0>, <2 x double>*@v2f64
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; ALL: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 1.0, double 31.0>, <2 x double>*@v2f64
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; ALL: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 3.0, double 4.0>, <2 x double>*@v2f64
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst($
  ; ALL: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
}

define void @nonconst_v4f32() nounwind {
  ; ALL-LABEL: nonconst_v4f32:

  %1 = load float , float *@f32
  %2 = insertelement <4 x float> undef, float %1, i32 0
  %3 = insertelement <4 x float> %2, float %1, i32 1
  %4 = insertelement <4 x float> %3, float %1, i32 2
  %5 = insertelement <4 x float> %4, float %1, i32 3
  store volatile <4 x float> %5, <4 x float>*@v4f32
  ; ALL: lwc1 $f[[R1:[0-9]+]], 0(
  ; ALL: splati.w [[R2:\$w[0-9]+]], $w[[R1]]

  ret void
}

define void @nonconst_v2f64() nounwind {
  ; ALL-LABEL: nonconst_v2f64:

  %1 = load double , double *@f64
  %2 = insertelement <2 x double> undef, double %1, i32 0
  %3 = insertelement <2 x double> %2, double %1, i32 1
  store volatile <2 x double> %3, <2 x double>*@v2f64
  ; ALL: ldc1 $f[[R1:[0-9]+]], 0(
  ; ALL: splati.d [[R2:\$w[0-9]+]], $w[[R1]]

  ret void
}

define float @extract_v4f32() nounwind {
  ; ALL-LABEL: extract_v4f32:

  %1 = load <4 x float>, <4 x float>* @v4f32
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = fadd <4 x float> %1, %1
  ; ALL-DAG: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <4 x float> %2, i32 1
  ; Element 1 can be obtained by splatting it across the vector and extracting
  ; $w0:sub_lo
  ; ALL-DAG: splati.w $w0, [[R1]][1]

  ret float %3
}

define float @extract_v4f32_elt0() nounwind {
  ; ALL-LABEL: extract_v4f32_elt0:

  %1 = load <4 x float>, <4 x float>* @v4f32
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = fadd <4 x float> %1, %1
  ; ALL-DAG: fadd.w $w0, [[R1]], [[R1]]

  %3 = extractelement <4 x float> %2, i32 0
  ; Element 0 can be obtained by extracting $w0:sub_lo ($f0)
  ; ALL-NOT: copy_u.w
  ; ALL-NOT: mtc1

  ret float %3
}

define float @extract_v4f32_elt2() nounwind {
  ; ALL-LABEL: extract_v4f32_elt2:

  %1 = load <4 x float>, <4 x float>* @v4f32
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = fadd <4 x float> %1, %1
  ; ALL-DAG: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <4 x float> %2, i32 2
  ; Element 2 can be obtained by splatting it across the vector and extracting
  ; $w0:sub_lo
  ; ALL-DAG: splati.w $w0, [[R1]][2]

  ret float %3
}

define float @extract_v4f32_vidx() nounwind {
  ; ALL-LABEL: extract_v4f32_vidx:

  %1 = load <4 x float>, <4 x float>* @v4f32
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v4f32)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v4f32)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v4f32)(
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = fadd <4 x float> %1, %1
  ; ALL-DAG: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <4 x float> %2, i32 %3
  ; ALL-DAG: splat.w $w0, [[R1]]{{\[}}[[IDX]]]

  ret float %4
}

define double @extract_v2f64() nounwind {
  ; ALL-LABEL: extract_v2f64:

  %1 = load <2 x double>, <2 x double>* @v2f64
  ; ALL-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = fadd <2 x double> %1, %1
  ; ALL-DAG: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <2 x double> %2, i32 1
  ; Element 1 can be obtained by splatting it across the vector and extracting
  ; $w0:sub_64
  ; ALL-DAG: splati.d $w0, [[R1]][1]
  ; ALL-NOT: copy_u.w
  ; ALL-NOT: mtc1
  ; ALL-NOT: mthc1
  ; ALL-NOT: sll
  ; ALL-NOT: sra

  ret double %3
}

define double @extract_v2f64_elt0() nounwind {
  ; ALL-LABEL: extract_v2f64_elt0:

  %1 = load <2 x double>, <2 x double>* @v2f64
  ; ALL-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = fadd <2 x double> %1, %1
  ; ALL-DAG: fadd.d $w0, [[R1]], [[R1]]

  %3 = extractelement <2 x double> %2, i32 0
  ; Element 0 can be obtained by extracting $w0:sub_64 ($f0)
  ; ALL-NOT: copy_u.w
  ; ALL-NOT: mtc1
  ; ALL-NOT: mthc1
  ; ALL-NOT: sll
  ; ALL-NOT: sra

  ret double %3
}

define double @extract_v2f64_vidx() nounwind {
  ; ALL-LABEL: extract_v2f64_vidx:

  %1 = load <2 x double>, <2 x double>* @v2f64
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v2f64)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v2f64)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v2f64)(
  ; ALL-DAG: ld.d [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = fadd <2 x double> %1, %1
  ; ALL-DAG: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <2 x double> %2, i32 %3
  ; ALL-DAG: splat.d $w0, [[R1]]{{\[}}[[IDX]]]

  ret double %4
}

define void @insert_v4f32(float %a) nounwind {
  ; ALL-LABEL: insert_v4f32:

  %1 = load <4 x float>, <4 x float>* @v4f32
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = insertelement <4 x float> %1, float %a, i32 1
  ; float argument passed in $f12
  ; ALL-DAG: insve.w [[R1]][1], $w12[0]

  store <4 x float> %2, <4 x float>* @v4f32
  ; ALL-DAG: st.w [[R1]]

  ret void
}

define void @insert_v2f64(double %a) nounwind {
  ; ALL-LABEL: insert_v2f64:

  %1 = load <2 x double>, <2 x double>* @v2f64
  ; ALL-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = insertelement <2 x double> %1, double %a, i32 1
  ; double argument passed in $f12
  ; ALL-DAG: insve.d [[R1]][1], $w12[0]

  store <2 x double> %2, <2 x double>* @v2f64
  ; ALL-DAG: st.d [[R1]]

  ret void
}

define void @insert_v4f32_vidx(float %a) nounwind {
  ; ALL-LABEL: insert_v4f32_vidx:

  %1 = load <4 x float>, <4 x float>* @v4f32
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v4f32)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v4f32)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v4f32)(
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %3 = insertelement <4 x float> %1, float %a, i32 %2
  ; float argument passed in $f12
  ; ALL-DAG: sll [[BIDX:\$[0-9]+]], [[IDX]], 2
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[BIDX]]]
  ; ALL-DAG: insve.w [[R1]][0], $w12[0]
  ; ALL-DAG: neg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[NIDX]]]

  store <4 x float> %3, <4 x float>* @v4f32
  ; ALL-DAG: st.w [[R1]]

  ret void
}

define void @insert_v2f64_vidx(double %a) nounwind {
  ; ALL-LABEL: insert_v2f64_vidx:

  %1 = load <2 x double>, <2 x double>* @v2f64
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v2f64)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v2f64)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v2f64)(
  ; ALL-DAG: ld.d [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %3 = insertelement <2 x double> %1, double %a, i32 %2
  ; double argument passed in $f12
  ; ALL-DAG: sll [[BIDX:\$[0-9]+]], [[IDX]], 3
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[BIDX]]]
  ; ALL-DAG: insve.d [[R1]][0], $w12[0]
  ; ALL-DAG: neg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[NIDX]]]

  store <2 x double> %3, <2 x double>* @v2f64
  ; ALL-DAG: st.d [[R1]]

  ret void
}
