;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=fiji -mattr=dumpcode --filetype=obj < %s  | llvm-objdump --triple=amdgcn --mcpu=fiji -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx900 -mattr=dumpcode --filetype=obj < %s  | llvm-objdump --triple=amdgcn --mcpu=gfx900 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx1010 -mattr=-real-true16 -mattr=dumpcode --filetype=obj < %s  | llvm-objdump --triple=amdgcn --mcpu=gfx1010 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx1100 -mattr=-real-true16 -amdgpu-enable-delay-alu=0 -mattr=dumpcode --filetype=obj < %s  | llvm-objdump --triple=amdgcn --mcpu=gfx1100 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

define i8 @v_saddsat_i8(i8 %lhs, i8 %rhs) {
  %result = call i8 @llvm.sadd.sat.i8(i8 %lhs, i8 %rhs)
  ret i8 %result
}

define i16 @v_saddsat_i16(i16 %lhs, i16 %rhs) {
  %result = call i16 @llvm.sadd.sat.i16(i16 %lhs, i16 %rhs)
  ret i16 %result
}

define i32 @v_saddsat_i32(i32 %lhs, i32 %rhs) {
  %result = call i32 @llvm.sadd.sat.i32(i32 %lhs, i32 %rhs)
  ret i32 %result
}

define <2 x i16> @v_saddsat_v2i16(<2 x i16> %lhs, <2 x i16> %rhs) {
  %result = call <2 x i16> @llvm.sadd.sat.v2i16(<2 x i16> %lhs, <2 x i16> %rhs)
  ret <2 x i16> %result
}

define <3 x i16> @v_saddsat_v3i16(<3 x i16> %lhs, <3 x i16> %rhs) {
  %result = call <3 x i16> @llvm.sadd.sat.v3i16(<3 x i16> %lhs, <3 x i16> %rhs)
  ret <3 x i16> %result
}

define <2 x float> @v_saddsat_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
  %result = call <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
  %cast = bitcast <4 x i16> %result to <2 x float>
  ret <2 x float> %cast
}

define <2 x i32> @v_saddsat_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
  %result = call <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
  ret <2 x i32> %result
}

define i64 @v_saddsat_i64(i64 %lhs, i64 %rhs) {
  %result = call i64 @llvm.sadd.sat.i64(i64 %lhs, i64 %rhs)
  ret i64 %result
}

declare i8 @llvm.sadd.sat.i8(i8, i8) #0
declare i16 @llvm.sadd.sat.i16(i16, i16) #0
declare <2 x i16> @llvm.sadd.sat.v2i16(<2 x i16>, <2 x i16>) #0
declare <3 x i16> @llvm.sadd.sat.v3i16(<3 x i16>, <3 x i16>) #0
declare <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16>, <4 x i16>) #0
declare i32 @llvm.sadd.sat.i32(i32, i32) #0
declare <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32>, <2 x i32>) #0
declare i64 @llvm.sadd.sat.i64(i64, i64) #0
