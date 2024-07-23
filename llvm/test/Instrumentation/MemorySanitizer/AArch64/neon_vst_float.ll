; No assertions yet because the test case crashes MSan
;
; Test memory sanitizer instrumentation for Arm NEON VST_{2,3,4} and
; VST_1x{2,3,4} instructions, including floating-point parameters.
;
; RUN: opt < %s -passes=msan -S | FileCheck %s
;
; UNSUPPORTED: {{.*}}
;
; Generated with:
;     grep call clang/test/CodeGen/aarch64-neon-intrinsics.c \
;         |  grep 'neon[.]st'                                \
;         | sed -r 's/^\/\/ CHECK:[ ]*//'                    \
;         | cut -d ' ' -f 1 --complement                     \
;         | sed -r 's/[[][[]TMP[0-9]+[]][]]/%A/'             \
;         | sed -r 's/[[][[]TMP[0-9]+[]][]]/%B/'             \
;         | sed -r 's/[[][[]TMP[0-9]+[]][]]/%C/'             \
;         | sed -r 's/[[][[]TMP[0-9]+[]][]]/%D/'             \
;         | sort                                             \
;         | uniq                                             \
;         | while read x;                                    \
;             do                                             \
;                 y=`echo "$x"                               \
;                     | sed -r 's/@llvm[.]aarch64[.]neon[.]/@/' \
;                     | sed -r 's/[.]p0//'                      \
;                     | tr '.' '_'`;                            \
;                 echo "define $y sanitize_memory {"; \
;                 echo "  call $x";                   \
;                 echo "  ret void";                  \
;                 echo "}";                           \
;                 echo;                               \
;             done

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android9001"

; -----------------------------------------------------------------------------------------------------------------------------------------------

define void @st1x2_v1f64(<1 x double> %A, <1 x double> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x2.v1f64.p0(<1 x double> %A, <1 x double> %B, ptr %a)
  ret void
}

define void @st1x2_v1i64(<1 x i64> %A, <1 x i64> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64> %A, <1 x i64> %B, ptr %a)
  ret void
}

define void @st1x2_v2f64(<2 x double> %A, <2 x double> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x2.v2f64.p0(<2 x double> %A, <2 x double> %B, ptr %a)
  ret void
}

define void @st1x2_v2i64(<2 x i64> %A, <2 x i64> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64> %A, <2 x i64> %B, ptr %a)
  ret void
}

define void @st1x3_v1f64(<1 x double> %A, <1 x double> %B, <1 x double> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x3.v1f64.p0(<1 x double> %A, <1 x double> %B, <1 x double> %C, ptr %a)
  ret void
}

define void @st1x3_v1i64(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, ptr %a)
  ret void
}

define void @st1x3_v2f64(<2 x double> %A, <2 x double> %B, <2 x double> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x3.v2f64.p0(<2 x double> %A, <2 x double> %B, <2 x double> %C, ptr %a)
  ret void
}

define void @st1x3_v2i64(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, ptr %a)
  ret void
}

define void @st1x4_v1f64(<1 x double> %A, <1 x double> %B, <1 x double> %C, <1 x double> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x4.v1f64.p0(<1 x double> %A, <1 x double> %B, <1 x double> %C, <1 x double> %D, ptr %a)
  ret void
}

define void @st1x4_v1i64(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, <1 x i64> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, <1 x i64> %D, ptr %a)
  ret void
}

define void @st1x4_v2f64(<2 x double> %A, <2 x double> %B, <2 x double> %C, <2 x double> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x4.v2f64.p0(<2 x double> %A, <2 x double> %B, <2 x double> %C, <2 x double> %D, ptr %a)
  ret void
}

define void @st1x4_v2i64(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, ptr %a)
  ret void
}

define void @st2_v16i8(<16 x i8> %A, <16 x i8> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> %A, <16 x i8> %B, ptr %a)
  ret void
}

define void @st2_v1f64(<1 x double> %A, <1 x double> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v1f64.p0(<1 x double> %A, <1 x double> %B, ptr %a)
  ret void
}

define void @st2_v1i64(<1 x i64> %A, <1 x i64> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> %A, <1 x i64> %B, ptr %a)
  ret void
}

define void @st2_v2f32(<2 x float> %A, <2 x float> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v2f32.p0(<2 x float> %A, <2 x float> %B, ptr %a)
  ret void
}

define void @st2_v2f64(<2 x double> %A, <2 x double> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v2f64.p0(<2 x double> %A, <2 x double> %B, ptr %a)
  ret void
}

define void @st2_v2i32(<2 x i32> %A, <2 x i32> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v2i32.p0(<2 x i32> %A, <2 x i32> %B, ptr %a)
  ret void
}

define void @st2_v2i64(<2 x i64> %A, <2 x i64> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> %A, <2 x i64> %B, ptr %a)
  ret void
}

define void @st2_v4f16(<4 x half> %A, <4 x half> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v4f16.p0(<4 x half> %A, <4 x half> %B, ptr %a)
  ret void
}

define void @st2_v4f32(<4 x float> %A, <4 x float> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v4f32.p0(<4 x float> %A, <4 x float> %B, ptr %a)
  ret void
}

define void @st2_v4i16(<4 x i16> %A, <4 x i16> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> %A, <4 x i16> %B, ptr %a)
  ret void
}

define void @st2_v4i32(<4 x i32> %A, <4 x i32> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %A, <4 x i32> %B, ptr %a)
  ret void
}

define void @st2_v8f16(<8 x half> %A, <8 x half> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v8f16.p0(<8 x half> %A, <8 x half> %B, ptr %a)
  ret void
}

define void @st2_v8i16(<8 x i16> %A, <8 x i16> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> %A, <8 x i16> %B, ptr %a)
  ret void
}

define void @st2_v8i8(<8 x i8> %A, <8 x i8> %B, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> %A, <8 x i8> %B, ptr %a)
  ret void
}

define void @st3_v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, ptr %a)
  ret void
}

define void @st3_v1f64(<1 x double> %A, <1 x double> %B, <1 x double> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v1f64.p0(<1 x double> %A, <1 x double> %B, <1 x double> %C, ptr %a)
  ret void
}

define void @st3_v1i64(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, ptr %a)
  ret void
}

define void @st3_v2f32(<2 x float> %A, <2 x float> %B, <2 x float> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v2f32.p0(<2 x float> %A, <2 x float> %B, <2 x float> %C, ptr %a)
  ret void
}

define void @st3_v2f64(<2 x double> %A, <2 x double> %B, <2 x double> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v2f64.p0(<2 x double> %A, <2 x double> %B, <2 x double> %C, ptr %a)
  ret void
}

define void @st3_v2i32(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v2i32.p0(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, ptr %a)
  ret void
}

define void @st3_v2i64(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, ptr %a)
  ret void
}

define void @st3_v4f16(<4 x half> %A, <4 x half> %B, <4 x half> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v4f16.p0(<4 x half> %A, <4 x half> %B, <4 x half> %C, ptr %a)
  ret void
}

define void @st3_v4f32(<4 x float> %A, <4 x float> %B, <4 x float> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v4f32.p0(<4 x float> %A, <4 x float> %B, <4 x float> %C, ptr %a)
  ret void
}

define void @st3_v4i16(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, ptr %a)
  ret void
}

define void @st3_v4i32(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, ptr %a)
  ret void
}

define void @st3_v8f16(<8 x half> %A, <8 x half> %B, <8 x half> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v8f16.p0(<8 x half> %A, <8 x half> %B, <8 x half> %C, ptr %a)
  ret void
}

define void @st3_v8i16(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, ptr %a)
  ret void
}

define void @st3_v8i8(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, ptr %a)
  ret void
}

define void @st4_v16i8(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> %A, <16 x i8> %B, <16 x i8> %C, <16 x i8> %D, ptr %a)
  ret void
}

define void @st4_v1f64(<1 x double> %A, <1 x double> %B, <1 x double> %C, <1 x double> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v1f64.p0(<1 x double> %A, <1 x double> %B, <1 x double> %C, <1 x double> %D, ptr %a)
  ret void
}

define void @st4_v1i64(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, <1 x i64> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> %A, <1 x i64> %B, <1 x i64> %C, <1 x i64> %D, ptr %a)
  ret void
}

define void @st4_v2f32(<2 x float> %A, <2 x float> %B, <2 x float> %C, <2 x float> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v2f32.p0(<2 x float> %A, <2 x float> %B, <2 x float> %C, <2 x float> %D, ptr %a)
  ret void
}

define void @st4_v2f64(<2 x double> %A, <2 x double> %B, <2 x double> %C, <2 x double> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v2f64.p0(<2 x double> %A, <2 x double> %B, <2 x double> %C, <2 x double> %D, ptr %a)
  ret void
}

define void @st4_v2i32(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v2i32.p0(<2 x i32> %A, <2 x i32> %B, <2 x i32> %C, <2 x i32> %D, ptr %a)
  ret void
}

define void @st4_v2i64(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> %A, <2 x i64> %B, <2 x i64> %C, <2 x i64> %D, ptr %a)
  ret void
}

define void @st4_v4f16(<4 x half> %A, <4 x half> %B, <4 x half> %C, <4 x half> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v4f16.p0(<4 x half> %A, <4 x half> %B, <4 x half> %C, <4 x half> %D, ptr %a)
  ret void
}

define void @st4_v4f32(<4 x float> %A, <4 x float> %B, <4 x float> %C, <4 x float> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v4f32.p0(<4 x float> %A, <4 x float> %B, <4 x float> %C, <4 x float> %D, ptr %a)
  ret void
}

define void @st4_v4i16(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, <4 x i16> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> %A, <4 x i16> %B, <4 x i16> %C, <4 x i16> %D, ptr %a)
  ret void
}

define void @st4_v4i32(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, <4 x i32> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v4i32.p0(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, <4 x i32> %D, ptr %a)
  ret void
}

define void @st4_v8f16(<8 x half> %A, <8 x half> %B, <8 x half> %C, <8 x half> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v8f16.p0(<8 x half> %A, <8 x half> %B, <8 x half> %C, <8 x half> %D, ptr %a)
  ret void
}

define void @st4_v8i16(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, <8 x i16> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> %A, <8 x i16> %B, <8 x i16> %C, <8 x i16> %D, ptr %a)
  ret void
}

define void @st4_v8i8(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, <8 x i8> %D, ptr %a) sanitize_memory {
  call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> %A, <8 x i8> %B, <8 x i8> %C, <8 x i8> %D, ptr %a)
  ret void
}
