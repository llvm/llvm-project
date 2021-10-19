; RUN: llc < %s -verify-machineinstrs -asm-verbose=false -mtriple=x86_64-unknown-linux-gnu -mcpu=nehalem -post-RA-scheduler=true -schedmodel=false | FileCheck %s

; Currently, floating-point selects are lowered to CFG triangles.
; This means that one side of the select is always unconditionally
; evaluated, however with MachineSink we can sink the other side so
; that it's conditionally evaluated.

; CHECK-LABEL: foo:
; CHECK-NEXT: testb $1, %dil
; CHECK-NEXT: jne
; CHECK-NEXT: divsd
; CHECK-NEXT: movapd
; CHECK-NEXT: ret
; CHECK:      divsd

define double @foo(double %x, double %y, i1 %c) nounwind {
  %a = fdiv double %x, 3.2
  %b = fdiv double %y, 3.3
  %z = select i1 %c, double %a, double %b
  ret double %z
}

; Make sure the critical edge is broken so the divsd is sunken below
; the conditional branch.
; rdar://8454886

; CHECK-LABEL: split:
; CHECK-NEXT: testb $1, %dil
; CHECK-NEXT: je
; CHECK:      divsd
; CHECK:      movapd
; CHECK:      ret
define double @split(double %x, double %y, i1 %c) nounwind {
  %a = fdiv double %x, 3.2
  %z = select i1 %c, double %a, double %y
  ret double %z
}


; Hoist floating-point constant-pool loads out of loops.

; CHECK-LABEL: bar:
; CHECK: movsd
; CHECK: align
define void @bar(double* nocapture %p, i64 %n) nounwind {
entry:
  %0 = icmp sgt i64 %n, 0
  br i1 %0, label %bb, label %return

bb:
  %i.03 = phi i64 [ 0, %entry ], [ %3, %bb ]
  %scevgep = getelementptr double, double* %p, i64 %i.03
  %1 = load double, double* %scevgep, align 8
  %2 = fdiv double 3.200000e+00, %1
  store double %2, double* %scevgep, align 8
  %3 = add nsw i64 %i.03, 1
  %exitcond = icmp eq i64 %3, %n
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; Sink instructions with dead EFLAGS defs.

; FIXME: Unfail the zzz test if we can correctly mark pregs with the kill flag.
;
; See <rdar://problem/8030636>. This test isn't valid after we made machine
; sinking more conservative about sinking instructions that define a preg into a
; block when we don't know if the preg is killed within the current block.


; FIXMEHECK: zzz:
; FIXMEHECK:      je
; FIXMEHECK-NEXT: orb

; define zeroext i8 @zzz(i8 zeroext %a, i8 zeroext %b) nounwind readnone {
; entry:
;   %tmp = zext i8 %a to i32                        ; <i32> [#uses=1]
;   %tmp2 = icmp eq i8 %a, 0                    ; <i1> [#uses=1]
;   %tmp3 = or i8 %b, -128                          ; <i8> [#uses=1]
;   %tmp4 = and i8 %b, 127                          ; <i8> [#uses=1]
;   %b_addr.0 = select i1 %tmp2, i8 %tmp4, i8 %tmp3 ; <i8> [#uses=1]
;   ret i8 %b_addr.0
; }

; Codegen should hoist and CSE these constants.

; CHECK-LABEL: vv:
; CHECK: LCPI3_0(%rip), %xmm0
; CHECK: LCPI3_1(%rip), %xmm1
; CHECK: LCPI3_2(%rip), %xmm2
; CHECK: align
; CHECK-NOT: LCPI
; CHECK: ret

@_minusZero.6007 = internal constant <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00> ; <<4 x float>*> [#uses=0]
@twoTo23.6008 = internal constant <4 x float> <float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06> ; <<4 x float>*> [#uses=0]

define void @vv(float* %y, float* %x, i32* %n) nounwind ssp {
entry:
  br label %bb60

bb:                                               ; preds = %bb60
  %i.0 = phi i32 [ 0, %bb60 ]                    ; <i32> [#uses=2]
  %0 = bitcast float* %x_addr.0 to <4 x float>*   ; <<4 x float>*> [#uses=1]
  %1 = load <4 x float>, <4 x float>* %0, align 16             ; <<4 x float>> [#uses=4]
  %tmp20 = bitcast <4 x float> %1 to <4 x i32>    ; <<4 x i32>> [#uses=1]
  %tmp22 = and <4 x i32> %tmp20, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647> ; <<4 x i32>> [#uses=1]
  %tmp23 = bitcast <4 x i32> %tmp22 to <4 x float> ; <<4 x float>> [#uses=1]
  %tmp25 = bitcast <4 x float> %1 to <4 x i32>    ; <<4 x i32>> [#uses=1]
  %tmp27 = and <4 x i32> %tmp25, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648> ; <<4 x i32>> [#uses=2]
  %tmp30 = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %tmp23, <4 x float> <float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06>, i8 5) ; <<4 x float>> [#uses=1]
  %tmp34 = bitcast <4 x float> %tmp30 to <4 x i32> ; <<4 x i32>> [#uses=1]
  %tmp36 = xor <4 x i32> %tmp34, <i32 -1, i32 -1, i32 -1, i32 -1> ; <<4 x i32>> [#uses=1]
  %tmp37 = and <4 x i32> %tmp36, <i32 1258291200, i32 1258291200, i32 1258291200, i32 1258291200> ; <<4 x i32>> [#uses=1]
  %tmp42 = or <4 x i32> %tmp37, %tmp27            ; <<4 x i32>> [#uses=1]
  %tmp43 = bitcast <4 x i32> %tmp42 to <4 x float> ; <<4 x float>> [#uses=2]
  %tmp45 = fadd <4 x float> %1, %tmp43            ; <<4 x float>> [#uses=1]
  %tmp47 = fsub <4 x float> %tmp45, %tmp43        ; <<4 x float>> [#uses=2]
  %tmp49 = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %1, <4 x float> %tmp47, i8 1) ; <<4 x float>> [#uses=1]
  %2 = bitcast <4 x float> %tmp49 to <4 x i32>    ; <<4 x i32>> [#uses=1]
  %3 = call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %2) nounwind readnone ; <<4 x float>> [#uses=1]
  %tmp53 = fadd <4 x float> %tmp47, %3            ; <<4 x float>> [#uses=1]
  %tmp55 = bitcast <4 x float> %tmp53 to <4 x i32> ; <<4 x i32>> [#uses=1]
  %tmp57 = or <4 x i32> %tmp55, %tmp27            ; <<4 x i32>> [#uses=1]
  %tmp58 = bitcast <4 x i32> %tmp57 to <4 x float> ; <<4 x float>> [#uses=1]
  %4 = bitcast float* %y_addr.0 to <4 x float>*   ; <<4 x float>*> [#uses=1]
  store <4 x float> %tmp58, <4 x float>* %4, align 16
  %5 = getelementptr float, float* %x_addr.0, i64 4      ; <float*> [#uses=1]
  %6 = getelementptr float, float* %y_addr.0, i64 4      ; <float*> [#uses=1]
  %7 = add i32 %i.0, 4                            ; <i32> [#uses=1]
  %8 = load i32, i32* %n, align 4                      ; <i32> [#uses=1]
  %9 = icmp sgt i32 %8, %7                        ; <i1> [#uses=1]
  br i1 %9, label %bb60, label %return

bb60:                                             ; preds = %bb, %entry
  %x_addr.0 = phi float* [ %x, %entry ], [ %5, %bb ] ; <float*> [#uses=2]
  %y_addr.0 = phi float* [ %y, %entry ], [ %6, %bb ] ; <float*> [#uses=2]
  br label %bb

return:                                           ; preds = %bb60
  ret void
}

declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

declare <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32>) nounwind readnone

; CodeGen should use the correct register class when extracting
; a load from a zero-extending load for hoisting.

; CHECK-LABEL: default_get_pch_validity:
; CHECK: movl cl_options_count(%rip), %ecx

@cl_options_count = external dso_local constant i32         ; <i32*> [#uses=2]

define void @default_get_pch_validity() nounwind {
entry:
  %tmp4 = load i32, i32* @cl_options_count, align 4    ; <i32> [#uses=1]
  %tmp5 = icmp eq i32 %tmp4, 0                    ; <i1> [#uses=1]
  br i1 %tmp5, label %bb6, label %bb2

bb2:                                              ; preds = %bb2, %entry
  %i.019 = phi i64 [ 0, %entry ], [ %tmp25, %bb2 ] ; <i64> [#uses=1]
  %tmp25 = add i64 %i.019, 1                      ; <i64> [#uses=2]
  %tmp11 = load i32, i32* @cl_options_count, align 4   ; <i32> [#uses=1]
  %tmp12 = zext i32 %tmp11 to i64                 ; <i64> [#uses=1]
  %tmp13 = icmp ugt i64 %tmp12, %tmp25            ; <i1> [#uses=1]
  br i1 %tmp13, label %bb2, label %bb6

bb6:                                              ; preds = %bb2, %entry
  ret void
}
