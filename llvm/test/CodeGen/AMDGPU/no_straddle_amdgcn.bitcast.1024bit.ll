;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=tonga  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=tonga -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx900  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=gfx900 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx1100  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=gfx1100 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

define <32 x float> @bitcast_v32i32_to_v32f32(<32 x i32> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define inreg <32 x float> @bitcast_v32i32_to_v32f32_scalar(<32 x i32> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define <32 x i32> @bitcast_v32f32_to_v32i32(<32 x float> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define inreg <32 x i32> @bitcast_v32f32_to_v32i32_scalar(<32 x float> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define <16 x i64> @bitcast_v32i32_to_v16i64(<32 x i32> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define inreg <16 x i64> @bitcast_v32i32_to_v16i64_scalar(<32 x i32> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define <32 x i32> @bitcast_v16i64_to_v32i32(<16 x i64> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define inreg <32 x i32> @bitcast_v16i64_to_v32i32_scalar(<16 x i64> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define <16 x double> @bitcast_v32i32_to_v16f64(<32 x i32> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define inreg <16 x double> @bitcast_v32i32_to_v16f64_scalar(<32 x i32> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define <32 x i32> @bitcast_v16f64_to_v32i32(<16 x double> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define inreg <32 x i32> @bitcast_v16f64_to_v32i32_scalar(<16 x double> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define <128 x i8> @bitcast_v32i32_to_v128i8(<32 x i32> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define inreg <128 x i8> @bitcast_v32i32_to_v128i8_scalar(<32 x i32> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define <32 x i32> @bitcast_v128i8_to_v32i32(<128 x i8> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define inreg <32 x i32> @bitcast_v128i8_to_v32i32_scalar(<128 x i8> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define <64 x bfloat> @bitcast_v32i32_to_v64bf16(<32 x i32> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define inreg <64 x bfloat> @bitcast_v32i32_to_v64bf16_scalar(<32 x i32> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define <32 x i32> @bitcast_v64bf16_to_v32i32(<64 x bfloat> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define inreg <32 x i32> @bitcast_v64bf16_to_v32i32_scalar(<64 x bfloat> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define <64 x half> @bitcast_v32i32_to_v64f16(<32 x i32> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define inreg <64 x half> @bitcast_v32i32_to_v64f16_scalar(<32 x i32> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define <32 x i32> @bitcast_v64f16_to_v32i32(<64 x half> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define inreg <32 x i32> @bitcast_v64f16_to_v32i32_scalar(<64 x half> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define <64 x i16> @bitcast_v32i32_to_v64i16(<32 x i32> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define inreg <64 x i16> @bitcast_v32i32_to_v64i16_scalar(<32 x i32> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <32 x i32> %a, splat (i32 3)
  %a2 = bitcast <32 x i32> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <32 x i32> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define <32 x i32> @bitcast_v64i16_to_v32i32(<64 x i16> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define inreg <32 x i32> @bitcast_v64i16_to_v32i32_scalar(<64 x i16> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <32 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <32 x i32>
  br label %end

end:
  %phi = phi <32 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x i32> %phi
}

define <16 x i64> @bitcast_v32f32_to_v16i64(<32 x float> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define inreg <16 x i64> @bitcast_v32f32_to_v16i64_scalar(<32 x float> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define <32 x float> @bitcast_v16i64_to_v32f32(<16 x i64> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define inreg <32 x float> @bitcast_v16i64_to_v32f32_scalar(<16 x i64> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define <16 x double> @bitcast_v32f32_to_v16f64(<32 x float> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define inreg <16 x double> @bitcast_v32f32_to_v16f64_scalar(<32 x float> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define <32 x float> @bitcast_v16f64_to_v32f32(<16 x double> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define inreg <32 x float> @bitcast_v16f64_to_v32f32_scalar(<16 x double> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define <128 x i8> @bitcast_v32f32_to_v128i8(<32 x float> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define inreg <128 x i8> @bitcast_v32f32_to_v128i8_scalar(<32 x float> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define <32 x float> @bitcast_v128i8_to_v32f32(<128 x i8> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define inreg <32 x float> @bitcast_v128i8_to_v32f32_scalar(<128 x i8> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define <64 x bfloat> @bitcast_v32f32_to_v64bf16(<32 x float> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define inreg <64 x bfloat> @bitcast_v32f32_to_v64bf16_scalar(<32 x float> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define <32 x float> @bitcast_v64bf16_to_v32f32(<64 x bfloat> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define inreg <32 x float> @bitcast_v64bf16_to_v32f32_scalar(<64 x bfloat> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define <64 x half> @bitcast_v32f32_to_v64f16(<32 x float> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define inreg <64 x half> @bitcast_v32f32_to_v64f16_scalar(<32 x float> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define <32 x float> @bitcast_v64f16_to_v32f32(<64 x half> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define inreg <32 x float> @bitcast_v64f16_to_v32f32_scalar(<64 x half> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define <64 x i16> @bitcast_v32f32_to_v64i16(<32 x float> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define inreg <64 x i16> @bitcast_v32f32_to_v64i16_scalar(<32 x float> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <32 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <32 x float> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <32 x float> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define <32 x float> @bitcast_v64i16_to_v32f32(<64 x i16> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define inreg <32 x float> @bitcast_v64i16_to_v32f32_scalar(<64 x i16> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <32 x float>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <32 x float>
  br label %end

end:
  %phi = phi <32 x float> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <32 x float> %phi
}

define <16 x double> @bitcast_v16i64_to_v16f64(<16 x i64> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define inreg <16 x double> @bitcast_v16i64_to_v16f64_scalar(<16 x i64> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define <16 x i64> @bitcast_v16f64_to_v16i64(<16 x double> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define inreg <16 x i64> @bitcast_v16f64_to_v16i64_scalar(<16 x double> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define <128 x i8> @bitcast_v16i64_to_v128i8(<16 x i64> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define inreg <128 x i8> @bitcast_v16i64_to_v128i8_scalar(<16 x i64> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define <16 x i64> @bitcast_v128i8_to_v16i64(<128 x i8> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define inreg <16 x i64> @bitcast_v128i8_to_v16i64_scalar(<128 x i8> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define <64 x bfloat> @bitcast_v16i64_to_v64bf16(<16 x i64> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define inreg <64 x bfloat> @bitcast_v16i64_to_v64bf16_scalar(<16 x i64> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define <16 x i64> @bitcast_v64bf16_to_v16i64(<64 x bfloat> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define inreg <16 x i64> @bitcast_v64bf16_to_v16i64_scalar(<64 x bfloat> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define <64 x half> @bitcast_v16i64_to_v64f16(<16 x i64> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define inreg <64 x half> @bitcast_v16i64_to_v64f16_scalar(<16 x i64> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define <16 x i64> @bitcast_v64f16_to_v16i64(<64 x half> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define inreg <16 x i64> @bitcast_v64f16_to_v16i64_scalar(<64 x half> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define <64 x i16> @bitcast_v16i64_to_v64i16(<16 x i64> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define inreg <64 x i16> @bitcast_v16i64_to_v64i16_scalar(<16 x i64> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <16 x i64> %a, splat (i64 3)
  %a2 = bitcast <16 x i64> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <16 x i64> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define <16 x i64> @bitcast_v64i16_to_v16i64(<64 x i16> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define inreg <16 x i64> @bitcast_v64i16_to_v16i64_scalar(<64 x i16> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <16 x i64>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <16 x i64>
  br label %end

end:
  %phi = phi <16 x i64> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x i64> %phi
}

define <128 x i8> @bitcast_v16f64_to_v128i8(<16 x double> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define inreg <128 x i8> @bitcast_v16f64_to_v128i8_scalar(<16 x double> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define <16 x double> @bitcast_v128i8_to_v16f64(<128 x i8> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define inreg <16 x double> @bitcast_v128i8_to_v16f64_scalar(<128 x i8> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define <64 x bfloat> @bitcast_v16f64_to_v64bf16(<16 x double> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define inreg <64 x bfloat> @bitcast_v16f64_to_v64bf16_scalar(<16 x double> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define <16 x double> @bitcast_v64bf16_to_v16f64(<64 x bfloat> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define inreg <16 x double> @bitcast_v64bf16_to_v16f64_scalar(<64 x bfloat> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define <64 x half> @bitcast_v16f64_to_v64f16(<16 x double> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define inreg <64 x half> @bitcast_v16f64_to_v64f16_scalar(<16 x double> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define <16 x double> @bitcast_v64f16_to_v16f64(<64 x half> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define inreg <16 x double> @bitcast_v64f16_to_v16f64_scalar(<64 x half> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define <64 x i16> @bitcast_v16f64_to_v64i16(<16 x double> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define inreg <64 x i16> @bitcast_v16f64_to_v64i16_scalar(<16 x double> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <16 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <16 x double> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <16 x double> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define <16 x double> @bitcast_v64i16_to_v16f64(<64 x i16> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define inreg <16 x double> @bitcast_v64i16_to_v16f64_scalar(<64 x i16> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <16 x double>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <16 x double>
  br label %end

end:
  %phi = phi <16 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <16 x double> %phi
}

define <64 x bfloat> @bitcast_v128i8_to_v64bf16(<128 x i8> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define inreg <64 x bfloat> @bitcast_v128i8_to_v64bf16_scalar(<128 x i8> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define <128 x i8> @bitcast_v64bf16_to_v128i8(<64 x bfloat> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define inreg <128 x i8> @bitcast_v64bf16_to_v128i8_scalar(<64 x bfloat> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define <64 x half> @bitcast_v128i8_to_v64f16(<128 x i8> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define inreg <64 x half> @bitcast_v128i8_to_v64f16_scalar(<128 x i8> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define <128 x i8> @bitcast_v64f16_to_v128i8(<64 x half> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define inreg <128 x i8> @bitcast_v64f16_to_v128i8_scalar(<64 x half> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define <64 x i16> @bitcast_v128i8_to_v64i16(<128 x i8> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define inreg <64 x i16> @bitcast_v128i8_to_v64i16_scalar(<128 x i8> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <128 x i8> %a, splat (i8 3)
  %a2 = bitcast <128 x i8> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <128 x i8> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define <128 x i8> @bitcast_v64i16_to_v128i8(<64 x i16> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define inreg <128 x i8> @bitcast_v64i16_to_v128i8_scalar(<64 x i16> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <128 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <128 x i8>
  br label %end

end:
  %phi = phi <128 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <128 x i8> %phi
}

define <64 x half> @bitcast_v64bf16_to_v64f16(<64 x bfloat> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define inreg <64 x half> @bitcast_v64bf16_to_v64f16_scalar(<64 x bfloat> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define <64 x bfloat> @bitcast_v64f16_to_v64bf16(<64 x half> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define inreg <64 x bfloat> @bitcast_v64f16_to_v64bf16_scalar(<64 x half> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define <64 x i16> @bitcast_v64bf16_to_v64i16(<64 x bfloat> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define inreg <64 x i16> @bitcast_v64bf16_to_v64i16_scalar(<64 x bfloat> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x bfloat> %a, splat (bfloat 0xR40C0)
  %a2 = bitcast <64 x bfloat> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <64 x bfloat> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define <64 x bfloat> @bitcast_v64i16_to_v64bf16(<64 x i16> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define inreg <64 x bfloat> @bitcast_v64i16_to_v64bf16_scalar(<64 x i16> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <64 x bfloat>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <64 x bfloat>
  br label %end

end:
  %phi = phi <64 x bfloat> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x bfloat> %phi
}

define <64 x i16> @bitcast_v64f16_to_v64i16(<64 x half> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define inreg <64 x i16> @bitcast_v64f16_to_v64i16_scalar(<64 x half> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <64 x half> %a, splat (half 0xH0200)
  %a2 = bitcast <64 x half> %a1 to <64 x i16>
  br label %end

cmp.false:
  %a3 = bitcast <64 x half> %a to <64 x i16>
  br label %end

end:
  %phi = phi <64 x i16> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x i16> %phi
}

define <64 x half> @bitcast_v64i16_to_v64f16(<64 x i16> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}

define inreg <64 x half> @bitcast_v64i16_to_v64f16_scalar(<64 x i16> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <64 x i16> %a, splat (i16 3)
  %a2 = bitcast <64 x i16> %a1 to <64 x half>
  br label %end

cmp.false:
  %a3 = bitcast <64 x i16> %a to <64 x half>
  br label %end

end:
  %phi = phi <64 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <64 x half> %phi
}
