;REQUIRES: asserts
;RUN: not --crash llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix < %s 2>&1 | FileCheck %s

;CHECK: %[[PHI1:[0-9]+]]:accrc = PHI
;CHECK: %[[PHI0:[0-9]+]]:uaccrc = PHI
;CHECK: Virtual register %[[PHI0]] is not needed live through the block.
;CHECK: Virtual register %[[PHI1]] must be live through the block.

target datalayout = "E-m:a-Fi64-i64:64-n32:64-S128-v256:256:256-v512:512:512"

define void @baz(i64 %arg) local_unnamed_addr #0 {
bb:
  %call = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.ppc.mma.disassemble.acc(<512 x i1> poison)
  %extractvalue = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %call, 0
  %extractvalue1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %call, 2
  %extractvalue2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %call, 3
  %bitcast = bitcast <16 x i8> %extractvalue to <2 x double>
  %bitcast3 = bitcast <16 x i8> %extractvalue1 to <2 x double>
  %shufflevector = shufflevector <2 x double> %bitcast, <2 x double> %bitcast3, <2 x i32> <i32 1, i32 3>
  %shufflevector4 = shufflevector <2 x double> %shufflevector, <2 x double> poison, <4 x i32> <i32 0, i32 poison, i32 1, i32 poison>
  %fneg = fneg <4 x double> %shufflevector4
  %shufflevector5 = shufflevector <4 x double> %fneg, <4 x double> poison, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  %shufflevector6 = shufflevector <16 x i8> zeroinitializer, <16 x i8> %extractvalue2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %bitcast7 = bitcast <32 x i8> %shufflevector6 to <4 x double>
  %fsub = fsub <4 x double> %shufflevector5, %bitcast7
  %shufflevector8 = shufflevector <4 x double> poison, <4 x double> %fsub, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  %fmul = fmul <4 x double> %shufflevector8, zeroinitializer
  %call9 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> zeroinitializer, <4 x double> zeroinitializer, <4 x double> %fmul)
  %fadd = fadd <4 x double> %call9, zeroinitializer
  store <4 x double> %fadd, ptr poison, align 4
  br i1 poison, label %bb10, label %bb12

bb10:                                             ; preds = %bb
  %call11 = tail call <512 x i1> @llvm.ppc.mma.xvf64gerpp(<512 x i1> poison, <256 x i1> poison, <16 x i8> poison)
  br label %bb12

bb12:                                             ; preds = %bb10, %bb
  %phi = phi <512 x i1> [ poison, %bb ], [ %call11, %bb10 ]
  br label %bb13

bb13:                                             ; preds = %bb13, %bb12
  %icmp = icmp eq i64 0, %arg
  br i1 %icmp, label %bb14, label %bb13

bb14:                                             ; preds = %bb13
  %call15 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.ppc.mma.disassemble.acc(<512 x i1> poison)
  %extractvalue16 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %call15, 1
  %call17 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.ppc.mma.disassemble.acc(<512 x i1> %phi)
  %extractvalue18 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %call17, 1
  %bitcast19 = bitcast <16 x i8> %extractvalue16 to <2 x double>
  %bitcast20 = bitcast <16 x i8> %extractvalue18 to <2 x double>
  %fsub21 = fsub <2 x double> zeroinitializer, %bitcast19
  %fadd22 = fadd <2 x double> zeroinitializer, %fsub21
  %fadd23 = fadd <2 x double> %fadd22, zeroinitializer
  %fsub24 = fsub <2 x double> zeroinitializer, %bitcast20
  %fadd25 = fadd <2 x double> zeroinitializer, %fsub24
  %fadd26 = fadd <2 x double> %fadd25, zeroinitializer
  %fadd27 = fadd <2 x double> %fadd26, zeroinitializer
  %call28 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %fadd27, <2 x double> zeroinitializer, <2 x double> zeroinitializer)
  %fadd29 = fadd <2 x double> zeroinitializer, %call28
  %shufflevector30 = shufflevector <2 x double> %fadd29, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  store <2 x double> %shufflevector30, ptr poison, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <512 x i1> @llvm.ppc.mma.xxsetaccz() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <512 x i1> @llvm.ppc.mma.xvf64gerpp(<512 x i1>, <256 x i1>, <16 x i8>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.ppc.mma.disassemble.acc(<512 x i1>) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #2

attributes #0 = { "target-features"="+altivec,+bpermd,+crbits,+crypto,+direct-move,+extdiv,+isa-v206-instructions,+isa-v207-instructions,+isa-v30-instructions,+isa-v31-instructions,+mma,+paired-vector-memops,+pcrelative-memops,+power10-vector,+power8-vector,+power9-vector,+prefix-instrs,+quadword-atomics,+vsx,-aix-shared-lib-tls-model-opt,-aix-small-local-dynamic-tls,-aix-small-local-exec-tls,-htm,-privileged,-rop-protect,-spe" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
