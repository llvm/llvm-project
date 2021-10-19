; RUN: llc -march=amdgcn -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89,GFX9,GFX9-F32FLUSH %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -denormal-fp-math-f32=ieee -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89,GFX9,GFX9-F32DENORM %s
; RUN: llc -march=amdgcn -mcpu=gfx803 -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89 %s
; RUN: llc -march=amdgcn -mcpu=gfx803 -denormal-fp-math-f32=ieee -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89 %s

;  fold (fadd (fpext (fmul x, y)), z) -> (fma (fpext x), (fpext y), z)

; GCN-LABEL: {{^}}fadd_fpext_fmul_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,0]{{$}}
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16
; GFX9-F32DENORM-NEXT: v_add_f32
define float @fadd_fpext_fmul_f16_to_f32(half %x, half %y, float %z) #0 {
entry:
  %mul = fmul half %x, %y
  %mul.ext = fpext half %mul to float
  %add = fadd float %mul.ext, %z
  ret float %add
}

; f16->f64 is not free.
; GCN-LABEL: {{^}}fadd_fpext_fmul_f16_to_f64:
; GFX89: v_mul_f16
; GFX89: v_cvt_f32_f16
; GFX89: v_cvt_f64_f32
; GFX89: v_add_f64
define double @fadd_fpext_fmul_f16_to_f64(half %x, half %y, double %z) #0 {
entry:
  %mul = fmul half %x, %y
  %mul.ext = fpext half %mul to double
  %add = fadd double %mul.ext, %z
  ret double %add
}

; f32->f64 is not free.
; GCN-LABEL: {{^}}fadd_fpext_fmul_f32_to_f64:
; GCN: v_mul_f32
; GCN: v_cvt_f64_f32
; GCN: v_add_f64
define double @fadd_fpext_fmul_f32_to_f64(float %x, float %y, double %z) #0 {
entry:
  %mul = fmul float %x, %y
  %mul.ext = fpext float %mul to double
  %add = fadd double %mul.ext, %z
  ret double %add
}

; fold (fadd x, (fpext (fmul y, z))) -> (fma (fpext y), (fpext z), x)
; GCN-LABEL: {{^}}fadd_fpext_fmul_f16_to_f32_commute:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,0]{{$}}
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16
; GFX9-F32DENORM-NEXT: v_add_f32
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fadd_fpext_fmul_f16_to_f32_commute(half %x, half %y, float %z) #0 {
entry:
  %mul = fmul half %x, %y
  %mul.ext = fpext half %mul to float
  %add = fadd float %z, %mul.ext
  ret float %add
}

; fold (fadd (fma x, y, (fpext (fmul u, v))), z)
;   -> (fma x, y, (fma (fpext u), (fpext v), z))

; GCN-LABEL: {{^}}fadd_muladd_fpext_fmul_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v2, v2, v3, v4 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: v_mac_f32_e32 v2, v0, v1
; GFX9-F32FLUSH-NEXT: v_mov_b32_e32 v0, v2
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16
; GFX9-F32DENORM-NEXT: v_fma_f32
; GFX9-F32DENORM-NEXT: v_add_f32
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fadd_muladd_fpext_fmul_f16_to_f32(float %x, float %y, half %u, half %v, float %z) #0 {
entry:
  %mul = fmul half %u, %v
  %mul.ext = fpext half %mul to float
  %fma = call float @llvm.fmuladd.f32(float %x, float %y, float %mul.ext)
  %add = fadd float %fma, %z
  ret float %add
}

; fold (fadd x, (fma y, z, (fpext (fmul u, v)))
;   -> (fma y, z, (fma (fpext u), (fpext v), x))
; GCN-LABEL: {{^}}fadd_muladd_fpext_fmul_f16_to_f32_commute:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v2, v2, v3, v4 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: v_mac_f32_e32 v2, v0, v1
; GFX9-F32FLUSH-NEXT: v_mov_b32_e32 v0, v2
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16
; GFX9-F32DENORM-NEXT: v_fma_f32
; GFX9-F32DENORM-NEXT: v_add_f32
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fadd_muladd_fpext_fmul_f16_to_f32_commute(float %x, float %y, half %u, half %v, float %z) #0 {
entry:
  %mul = fmul half %u, %v
  %mul.ext = fpext half %mul to float
  %fma = call float @llvm.fmuladd.f32(float %x, float %y, float %mul.ext)
  %add = fadd float %z, %fma
  ret float %add
}

; GCN-LABEL: {{^}}fadd_fmad_fpext_fmul_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v2, v2, v3, v4 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: v_mac_f32_e32 v2, v0, v1
; GFX9-F32FLUSH-NEXT: v_mov_b32_e32 v0, v2
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16_e32 v2, v2, v3
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16_e32 v2, v2
; GFX9-F32DENORM-NEXT: v_fma_f32 v0, v0, v1, v2
define float @fadd_fmad_fpext_fmul_f16_to_f32(float %x, float %y, half %u, half %v, float %z) #0 {
entry:
  %mul = fmul half %u, %v
  %mul.ext = fpext half %mul to float
  %mul1 = fmul contract float %x, %y
  %fmad = fadd contract float %mul1, %mul.ext
  %add = fadd float %fmad, %z
  ret float %add
}

; fold (fadd (fma x, y, (fpext (fmul u, v))), z)
;   -> (fma x, y, (fma (fpext u), (fpext v), z))

; GCN-LABEL: {{^}}fadd_fma_fpext_fmul_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v2, v2, v3, v4 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: v_mac_f32_e32 v2, v0, v1
; GFX9-F32FLUSH-NEXT: v_mov_b32_e32 v0, v2
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16_e32 v2, v2, v3
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16_e32 v2, v2
; GFX9-F32DENORM-NEXT: v_fma_f32 v0, v0, v1, v2
; GFX9-F32DENORM-NEXT: v_add_f32_e32 v0, v0, v4
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fadd_fma_fpext_fmul_f16_to_f32(float %x, float %y, half %u, half %v, float %z) #0 {
entry:
  %mul = fmul contract half %u, %v
  %mul.ext = fpext half %mul to float
  %fma = call float @llvm.fma.f32(float %x, float %y, float %mul.ext)
  %add = fadd float %fma, %z
  ret float %add
}

; GCN-LABEL: {{^}}fadd_fma_fpext_fmul_f16_to_f32_commute:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v2, v2, v3, v4 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: v_mac_f32_e32 v2, v0, v1
; GFX9-F32FLUSH-NEXT: v_mov_b32_e32 v0, v2
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16_e32 v2, v2, v3
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16_e32 v2, v2
; GFX9-F32DENORM-NEXT: v_fma_f32 v0, v0, v1, v2
; GFX9-F32DENORM-NEXT: v_add_f32_e32 v0, v4, v0
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fadd_fma_fpext_fmul_f16_to_f32_commute(float %x, float %y, half %u, half %v, float %z) #0 {
entry:
  %mul = fmul contract half %u, %v
  %mul.ext = fpext half %mul to float
  %fma = call float @llvm.fma.f32(float %x, float %y, float %mul.ext)
  %add = fadd float %z, %fma
  ret float %add
}

; fold (fadd x, (fpext (fma y, z, (fmul u, v)))
;   -> (fma (fpext y), (fpext z), (fma (fpext u), (fpext v), x))

; GCN-LABEL: {{^}}fadd_fpext_fmuladd_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v3, v4, v0 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v1, v2, v0 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16
; GFX9-F32DENORM-NEXT: v_fma_f16
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16
; GFX9-F32DENORM-NEXT: v_add_f32
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fadd_fpext_fmuladd_f16_to_f32(float %x, half %y, half %z, half %u, half %v) #0 {
entry:
  %mul = fmul contract half %u, %v
  %fma = call half @llvm.fmuladd.f16(half %y, half %z, half %mul)
  %ext.fma = fpext half %fma to float
  %add = fadd float %x, %ext.fma
  ret float %add
}

; GCN-LABEL: {{^}}fadd_fpext_fma_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v3, v4, v0 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v1, v2, v0 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16
; GFX9-F32DENORM-NEXT: v_fma_f16
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16
; GFX9-F32DENORM-NEXT: v_add_f32
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fadd_fpext_fma_f16_to_f32(float %x, half %y, half %z, half %u, half %v) #0 {
entry:
  %mul = fmul contract half %u, %v
  %fma = call half @llvm.fma.f16(half %y, half %z, half %mul)
  %ext.fma = fpext half %fma to float
  %add = fadd float %x, %ext.fma
  ret float %add
}

; GCN-LABEL: {{^}}fadd_fpext_fma_f16_to_f32_commute:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v3, v4, v0 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v1, v2, v0 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16
; GFX9-F32DENORM-NEXT: v_fma_f16
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16
; GFX9-F32DENORM-NEXT: v_add_f32_e32
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fadd_fpext_fma_f16_to_f32_commute(float %x, half %y, half %z, half %u, half %v) #0 {
entry:
  %mul = fmul contract half %u, %v
  %fma = call half @llvm.fma.f16(half %y, half %z, half %mul)
  %ext.fma = fpext half %fma to float
  %add = fadd float %ext.fma, %x
  ret float %add
}

; fold (fsub (fpext (fmul x, y)), z)
;   -> (fma (fpext x), (fpext y), (fneg z))

; GCN-LABEL: {{^}}fsub_fpext_fmul_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v0, v1, -v2 op_sel_hi:[1,1,0]{{$}}
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16_e32 v0, v0, v1
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16_e32 v0, v0
; GFX9-F32DENORM-NEXT: v_sub_f32_e32 v0, v0, v2
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fsub_fpext_fmul_f16_to_f32(half %x, half %y, float %z) #0 {
entry:
  %mul = fmul half %x, %y
  %mul.ext = fpext half %mul to float
  %add = fsub float %mul.ext, %z
  ret float %add
}

; fold (fsub x, (fpext (fmul y, z)))
;   -> (fma (fneg (fpext y)), (fpext z), x)

; GCN-LABEL: {{^}}fsub_fpext_fmul_f16_to_f32_commute:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, -v1, v2, v0 op_sel_hi:[1,1,0]
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16_e32
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16_e32
; GFX9-F32DENORM-NEXT: v_sub_f32_e32
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fsub_fpext_fmul_f16_to_f32_commute(float %x, half %y, half %z) #0 {
entry:
  %mul = fmul contract half %y, %z
  %mul.ext = fpext half %mul to float
  %add = fsub contract float %x, %mul.ext
  ret float %add
}

; fold (fsub (fpext (fneg (fmul, x, y))), z)
;   -> (fneg (fma (fpext x), (fpext y), z))

; GCN-LABEL: {{^}}fsub_fpext_fneg_fmul_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v0, -v1, -v2 op_sel_hi:[1,1,0]{{$}}
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16_e64 v0, v0, -v1
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16_e32 v0, v0
; GFX9-F32DENORM-NEXT: v_sub_f32_e32 v0, v0, v2
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fsub_fpext_fneg_fmul_f16_to_f32(half %x, half %y, float %z) #0 {
entry:
  %mul = fmul half %x, %y
  %neg.mul = fsub half -0.0, %mul
  %neg.mul.ext = fpext half %neg.mul to float
  %add = fsub float %neg.mul.ext, %z
  ret float %add
}

; fold (fsub (fneg (fpext (fmul, x, y))), z)
;   -> (fneg (fma (fpext x)), (fpext y), z)

; GCN-LABEL: {{^}}fsub_fneg_fpext_fmul_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, v0, -v1, -v2 op_sel_hi:[1,1,0]{{$}}
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16_e64 v0, v0, -v1
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16_e32 v0, v0
; GFX9-F32DENORM-NEXT: v_sub_f32_e32 v0, v0, v2
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fsub_fneg_fpext_fmul_f16_to_f32(half %x, half %y, float %z) #0 {
entry:
  %mul = fmul half %x, %y
  %mul.ext = fpext half %mul to float
  %neg.mul.ext = fneg float %mul.ext
  %add = fsub float %neg.mul.ext, %z
  ret float %add
}

; fold (fsub (fmad x, y, (fpext (fmul u, v))), z)
;    -> (fmad x, y (fmad (fpext u), (fpext v), (fneg z)))
; GCN-LABEL: {{^}}fsub_muladd_fpext_mul_f16_to_f32:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v2, v3, v4, -v2 op_sel_hi:[1,1,0]{{$}}
; GFX9-F32FLUSH-NEXT: v_mac_f32_e32 v2, v0, v1
; GFX9-F32FLUSH-NEXT: v_mov_b32_e32 v0, v2
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16_e32 v3, v3, v4
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16_e32 v3, v3
; GFX9-F32DENORM-NEXT: v_fma_f32 v0, v0, v1, v3
; GFX9-F32DENORM-NEXT: v_sub_f32_e32 v0, v0, v2
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fsub_muladd_fpext_mul_f16_to_f32(float %x, float %y, float %z, half %u, half %v) #0 {
entry:
  %mul = fmul reassoc half %u, %v
  %mul.ext = fpext half %mul to float
  %fma = call float @llvm.fmuladd.f32(float %x, float %y, float %mul.ext)
  %add = fsub reassoc float %fma, %z
  ret float %add
}

;  fold (fsub (fpext (fmad x, y, (fmul u, v))), z)
;    -> (fmad (fpext x), (fpext y),
;            (fmad (fpext u), (fpext v), (fneg z)))

; GCN-LABEL: {{^}}fsub_fpext_muladd_mul_f16_to_f32:
; GFX9: v_mul_f16
; GFX9: v_fma_f16
; GFX9: v_cvt_f32_f16
; GFX9: v_sub_f32
; GCN: s_setpc_b64
define float @fsub_fpext_muladd_mul_f16_to_f32(half %x, half %y, float %z, half %u, half %v) #0 {
entry:
  %mul = fmul half %u, %v
  %fma = call half @llvm.fmuladd.f16(half %x, half %y, half %mul)
  %fma.ext = fpext half %fma to float
  %add = fsub float %fma.ext, %z
  ret float %add
}

; fold (fsub x, (fmad y, z, (fpext (fmul u, v))))
;   -> (fmad (fneg y), z, (fmad (fneg (fpext u)), (fpext v), x))
; GCN-LABEL: {{^}}fsub_muladd_fpext_mul_f16_to_f32_commute:
; GCN: s_waitcnt
; GFX9-F32FLUSH-NEXT: v_mad_mix_f32 v0, -v3, v4, v0 op_sel_hi:[1,1,0]{{$}}
; GFX9-F32FLUSH-NEXT: v_mad_f32 v0, -v1, v2, v0{{$}}
; GFX9-F32FLUSH-NEXT: s_setpc_b64

; GFX9-F32DENORM-NEXT: v_mul_f16_e32 v3, v3, v4
; GFX9-F32DENORM-NEXT: v_cvt_f32_f16_e32 v3, v3
; GFX9-F32DENORM-NEXT: v_fma_f32 v1, v1, v2, v3
; GFX9-F32DENORM-NEXT: v_sub_f32_e32 v0, v0, v1
; GFX9-F32DENORM-NEXT: s_setpc_b64
define float @fsub_muladd_fpext_mul_f16_to_f32_commute(float %x, float %y, float %z, half %u, half %v) #0 {
entry:
  %mul = fmul reassoc half %u, %v
  %mul.ext = fpext half %mul to float
  %fma = call float @llvm.fmuladd.f32(float %y, float %z, float %mul.ext)
  %add = fsub reassoc float %x, %fma
  ret float %add
}

; fold (fsub x, (fpext (fma y, z, (fmul u, v))))
;    -> (fma (fneg (fpext y)), (fpext z),
;            (fma (fneg (fpext u)), (fpext v), x))
; GCN-LABEL: {{^}}fsub_fpext_muladd_mul_f16_to_f32_commute:
; GCN: s_waitcnt
; GFX9-NEXT: v_mul_f16_e32 v3, v3, v4
; GFX9-NEXT: v_fma_f16 v1, v1, v2, v3
; GFX9-NEXT: v_cvt_f32_f16_e32 v1, v1
; GFX9-NEXT: v_sub_f32_e32 v0, v0, v1
; GFX9-NEXT: s_setpc_b64
define float @fsub_fpext_muladd_mul_f16_to_f32_commute(float %x, half %y, half %z, half %u, half %v) #0 {
entry:
  %mul = fmul half %u, %v
  %fma = call half @llvm.fmuladd.f16(half %y, half %z, half %mul)
  %fma.ext = fpext half %fma to float
  %add = fsub float %x, %fma.ext
  ret float %add
}

declare float @llvm.fmuladd.f32(float, float, float) #0
declare float @llvm.fma.f32(float, float, float) #0
declare half @llvm.fmuladd.f16(half, half, half) #0
declare half @llvm.fma.f16(half, half, half) #0

attributes #0 = { nounwind readnone speculatable }
