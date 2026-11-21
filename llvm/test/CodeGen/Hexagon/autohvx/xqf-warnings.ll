; XFAIL: hexagon-registered-target
; Tests for emitted warnings when IEEE type is used as qf and vice-versa
; Test program source code with line number:

;1 #include <hexagon_types.h>
;2 HEXAGON_Vect2048 foo(HEXAGON_Vect1024 vina, HEXAGON_Vect1024 vinb) {
;3   const HEXAGON_Vect1024 ishf1 = __builtin_HEXAGON_V6_lvsplath_128B(0x3C00);
;4   const HEXAGON_Vect1024 ishf2 = __builtin_HEXAGON_V6_lvsplath_128B(0xBC00);
;5   const HEXAGON_Vect1024 issf1 = __builtin_HEXAGON_V6_lvsplatw_128B(0xAC00);
;6   const HEXAGON_Vect1024 issf2 = __builtin_HEXAGON_V6_lvsplatw_128B(0xDC00);
;7   const HEXAGON_Vect1024 isqf16 = __builtin_HEXAGON_V6_vadd_qf16_mix_128B(ishf1, ishf2);
;8   const HEXAGON_Vect1024 isqf32 = __builtin_HEXAGON_V6_vadd_qf32_mix_128B(issf1, issf2);
;9   HEXAGON_Vect2048 isqf32_1 = __builtin_HEXAGON_V6_vmpy_qf32_qf16_128B(vina,isqf16);
;10   HEXAGON_Vect2048 isqf32_2 = __builtin_HEXAGON_V6_vmpy_qf32_hf_128B(vinb,isqf16);
;11   HEXAGON_Vect1024 add1 = __builtin_HEXAGON_V6_vsub_qf32_mix_128B(__builtin_HEXAGON_V6_hi_128B(isqf32_1),isqf32);
;12   HEXAGON_Vect1024 add2 = __builtin_HEXAGON_V6_vsub_qf32_128B(__builtin_HEXAGON_V6_hi_128B(isqf32_2),isqf32);
;13   return __builtin_HEXAGON_V6_vcombine_128B(add2,add1);
;14 }

; RUN: llc --mtriple=hexagon-- -mhvx -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -verify-machineinstrs \
; RUN: -hexagon-qfloat-mode=ieee 2>&1 < %s -o /dev/null | FileCheck %s
; RUN: llc --mtriple=hexagon-- -mhvx -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b,+hvx-qfloat -enable-xqf-gen=true \
; RUN: -verify-machineinstrs \
; RUN: -hexagon-qfloat-mode=ieee 2>&1 < %s -o /dev/null | FileCheck %s

define dso_local inreg <64 x i32> @foo(<32 x i32> noundef %vina, <32 x i32> noundef %vinb) local_unnamed_addr #0 !dbg !8 {
; CHECK-NOT: warning: test.c:3:
; CHECK-NOT: warning: test.c:4:
; CHECK-NOT: warning: test.c:5:
; CHECK-NOT: warning: test.c:6:
; CHECK: warning: test.c:7:35: in function foo <64 x i32> (<32 x i32>, <32 x i32>): hf type used as qf16 at operand 1
; CHECK: warning: test.c:8:35: in function foo <64 x i32> (<32 x i32>, <32 x i32>): sf type used as qf32 at operand 1
; CHECK: warning: test.c:9:31: in function foo <64 x i32> (<32 x i32>, <32 x i32>): hf type used as qf16 at operand 1
; CHECK: warning: test.c:10:31: in function foo <64 x i32> (<32 x i32>, <32 x i32>): qf16 type used as hf at operand 2
; CHECK: warning: test.c:11:67: in function foo <64 x i32> (<32 x i32>, <32 x i32>): qf32 type used as sf at operand 1
; CHECK: warning: test.c:11:27: in function foo <64 x i32> (<32 x i32>, <32 x i32>): sf type used as qf32 at operand 1
; CHECK: warning: test.c:11:27: in function foo <64 x i32> (<32 x i32>, <32 x i32>): qf32 type used as sf at operand 2
; CHECK: warning: test.c:12:63: in function foo <64 x i32> (<32 x i32>, <32 x i32>): qf32 type used as sf at operand 1
; CHECK: warning: test.c:12:27: in function foo <64 x i32> (<32 x i32>, <32 x i32>): sf type used as qf32 at operand 1
; CHECK-NOT: warning: test.c:12: {{.*}}: qf32 type used as sf at operand 2
; CHECK-NOT: warning: test.c:12: {{.*}}: sf type used as qf32 at operand 2
; CHECK: warning: test.c:13:10: in function foo <64 x i32> (<32 x i32>, <32 x i32>): qf32 type used as sf at operand 1
; CHECK: warning: test.c:13:10: in function foo <64 x i32> (<32 x i32>, <32 x i32>): qf32 type used as sf at operand 2
entry:
  call void @llvm.dbg.value(metadata <32 x i32> %vina, metadata !22, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata <32 x i32> %vinb, metadata !23, metadata !DIExpression()), !dbg !35
  %0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 15360), !dbg !36
  call void @llvm.dbg.value(metadata <32 x i32> %0, metadata !24, metadata !DIExpression()), !dbg !35
  %1 = tail call <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32 48128), !dbg !37
  call void @llvm.dbg.value(metadata <32 x i32> %1, metadata !26, metadata !DIExpression()), !dbg !35
  %2 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 44032), !dbg !38
  call void @llvm.dbg.value(metadata <32 x i32> %2, metadata !27, metadata !DIExpression()), !dbg !35
  %3 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 56320), !dbg !39
  call void @llvm.dbg.value(metadata <32 x i32> %3, metadata !28, metadata !DIExpression()), !dbg !35
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32> %0, <32 x i32> %1), !dbg !40
  call void @llvm.dbg.value(metadata <32 x i32> %4, metadata !29, metadata !DIExpression()), !dbg !35
  %5 = tail call <32 x i32> @llvm.hexagon.V6.vadd.qf32.mix.128B(<32 x i32> %2, <32 x i32> %3), !dbg !41
  call void @llvm.dbg.value(metadata <32 x i32> %5, metadata !30, metadata !DIExpression()), !dbg !35
  %6 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.qf16.128B(<32 x i32> %vina, <32 x i32> %4), !dbg !42
  call void @llvm.dbg.value(metadata <64 x i32> %6, metadata !31, metadata !DIExpression()), !dbg !35
  %7 = tail call <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(<32 x i32> %vinb, <32 x i32> %4), !dbg !43
  call void @llvm.dbg.value(metadata <64 x i32> %7, metadata !32, metadata !DIExpression()), !dbg !35
  %8 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %6), !dbg !44
  %9 = tail call <32 x i32> @llvm.hexagon.V6.vsub.qf32.mix.128B(<32 x i32> %8, <32 x i32> %5), !dbg !45
  call void @llvm.dbg.value(metadata <32 x i32> %9, metadata !33, metadata !DIExpression()), !dbg !35
  %10 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %7), !dbg !46
  %11 = tail call <32 x i32> @llvm.hexagon.V6.vsub.qf32.128B(<32 x i32> %10, <32 x i32> %5), !dbg !47
  call void @llvm.dbg.value(metadata <32 x i32> %11, metadata !34, metadata !DIExpression()), !dbg !35
  %12 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %11, <32 x i32> %9), !dbg !48
  ret <64 x i32> %12, !dbg !49
}

declare <32 x i32> @llvm.hexagon.V6.lvsplath.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.qf16.mix.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadd.qf32.mix.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vmpy.qf32.qf16.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vmpy.qf32.hf.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsub.qf32.mix.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsub.qf32.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(none) "approx-func-fp-math"="true" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv79" "target-features"="+hvx-length128b,+hvx-qfloat,+hvxv79,+v79,-long-calls" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
