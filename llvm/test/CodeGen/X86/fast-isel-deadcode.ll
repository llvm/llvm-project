; RUN: llc < %s | FileCheck %s
;
; Generated with clang -O2 -S -emit-llvm
;
; /* Test 1 */
; extern "C" bool bar (long double);
; __attribute__((optnone))
; extern "C" bool foo(long double x, long double y)
; {
;   return (x == y) || (bar(x));
; }
;
; /* Test 2 */
; struct FVector {
;   float x, y, z;
;   inline __attribute__((always_inline)) FVector(float f): x(f), y(f), z(f) {}
;   inline __attribute__((always_inline)) FVector func(float p) const
;   {
;     if( x == 1.f ) {
;       return *this;
;     } else if( x < p ) {
;       return FVector(0.f);
;     }
;     return FVector(x);
;   }
; };
; 
; __attribute__((optnone))
; int main()
; {
;   FVector v(1.0);
;   v = v.func(1.e-8);
;   return 0;
; }
;
; ModuleID = 'test.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.FVector = type { float, float, float }

define zeroext i1 @foo(x86_fp80 %x, x86_fp80 %y) noinline optnone {
entry:
  %x.addr = alloca x86_fp80, align 16
  %y.addr = alloca x86_fp80, align 16
  store x86_fp80 %x, ptr %x.addr, align 16
  store x86_fp80 %y, ptr %y.addr, align 16
  %0 = load x86_fp80, ptr %x.addr, align 16
  %1 = load x86_fp80, ptr %y.addr, align 16
  %cmp = fcmp oeq x86_fp80 %0, %1

; Test 1
; Make sure that there is no dead code generated
; from Fast-ISel Phi-node handling. We should only
; see one movb of the constant 1, feeding the PHI
; node in lor.end. This covers the code path with
; handlePHINodesInSuccessorBlocks() returning true.
;
; CHECK-LABEL: foo:
; CHECK: movb $1,
; CHECK-NOT: movb $1,
; CHECK-LABEL: .LBB0_1:

  br i1 %cmp, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %entry
  %2 = load x86_fp80, ptr %x.addr, align 16
  %call = call zeroext i1 @bar(x86_fp80 %2)
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %3 = phi i1 [ true, %entry ], [ %call, %lor.rhs ]
  ret i1 %3
}

declare zeroext i1 @bar(x86_fp80)

define i32 @main() noinline optnone {
entry:
  %retval = alloca i32, align 4
  %v = alloca %struct.FVector, align 4
  %ref.tmp = alloca %struct.FVector, align 4
  %tmp = alloca { <2 x float>, float }, align 8
  store i32 0, ptr %retval, align 4
  call void @llvm.lifetime.start.p0(i64 12, ptr %v) nounwind
  store float 1.000000e+00, ptr %v, align 4
  %y.i = getelementptr inbounds %struct.FVector, ptr %v, i64 0, i32 1
  store float 1.000000e+00, ptr %y.i, align 4
  %z.i = getelementptr inbounds %struct.FVector, ptr %v, i64 0, i32 2
  store float 1.000000e+00, ptr %z.i, align 4
  %0 = load float, ptr %v, align 4
  %cmp.i = fcmp oeq float %0, 1.000000e+00
  br i1 %cmp.i, label %if.then.i, label %if.else.i

if.then.i:                                        ; preds = %entry
  %retval.sroa.0.0.copyload.i = load <2 x float>, ptr %v, align 4
  %retval.sroa.6.0..sroa_idx16.i = getelementptr inbounds %struct.FVector, ptr %v, i64 0, i32 2
  %retval.sroa.6.0.copyload.i = load float, ptr %retval.sroa.6.0..sroa_idx16.i, align 4
  br label %func.exit

if.else.i:                                        ; preds = %entry

; Test 2
; In order to feed the first PHI node in func.exit handlePHINodesInSuccessorBlocks()
; generates a local value instruction, but it cannot handle the second PHI node and
; returns false to let SelectionDAGISel handle both cases. Make sure the generated 
; local value instruction is removed.
; CHECK-LABEL: main:
; CHECK-LABEL: .LBB1_2:
; CHECK:       xorps [[REG:%xmm[0-7]]], [[REG]]
; CHECK-NOT:   xorps [[REG]], [[REG]]
; CHECK-LABEL: .LBB1_3:

  %cmp3.i = fcmp olt float %0, 0x3E45798EE0000000
  br i1 %cmp3.i, label %func.exit, label %if.end.5.i

if.end.5.i:                                       ; preds = %if.else.i
  %retval.sroa.0.0.vec.insert13.i = insertelement <2 x float> undef, float %0, i32 0
  %retval.sroa.0.4.vec.insert15.i = insertelement <2 x float> %retval.sroa.0.0.vec.insert13.i, float %0, i32 1
  br label %func.exit

func.exit:                         ; preds = %if.then.i, %if.else.i, %if.end.5.i
  %retval.sroa.6.0.i = phi float [ %retval.sroa.6.0.copyload.i, %if.then.i ], [ %0, %if.end.5.i ], [ 0.000000e+00, %if.else.i ]
  %retval.sroa.0.0.i = phi <2 x float> [ %retval.sroa.0.0.copyload.i, %if.then.i ], [ %retval.sroa.0.4.vec.insert15.i, %if.end.5.i ], [ zeroinitializer, %if.else.i ]
  %.fca.0.insert.i = insertvalue { <2 x float>, float } undef, <2 x float> %retval.sroa.0.0.i, 0
  %.fca.1.insert.i = insertvalue { <2 x float>, float } %.fca.0.insert.i, float %retval.sroa.6.0.i, 1
  store { <2 x float>, float } %.fca.1.insert.i, ptr %tmp, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %ref.tmp, ptr align 4 %tmp, i64 12, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %v, ptr align 4 %ref.tmp, i64 12, i1 false)
  call void @llvm.lifetime.end.p0(i64 12, ptr %v) nounwind
  ret i32 0
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) argmemonly nounwind

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) argmemonly nounwind

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) argmemonly nounwind
