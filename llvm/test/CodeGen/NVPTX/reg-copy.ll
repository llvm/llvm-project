; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

define ptx_kernel void @PR24303(ptr %f) {
; CHECK-LABEL: .visible .entry PR24303(
; Do not use mov.f or mov.u to convert between float and int.
; CHECK-NOT: mov.{{f|u}}{{32|64}} %f{{[0-9]+}}, %r{{[0-9]+}}
; CHECK-NOT: mov.{{f|u}}{{32|64}} %r{{[0-9]+}}, %f{{[0-9]+}}
entry:
  %arrayidx1 = getelementptr inbounds float, ptr %f, i64 1
  %0 = load float, ptr %f, align 4
  %1 = load float, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %f, i64 2
  %arrayidx3 = getelementptr inbounds float, ptr %f, i64 3
  %2 = load float, ptr %arrayidx2, align 4
  %3 = load float, ptr %arrayidx3, align 4
  %mul.i = fmul float %0, %2
  %mul4.i = fmul float %1, %3
  %mul5.i = fmul float %0, %3
  %mul6.i = fmul float %1, %2
  %sub.i = fsub float %mul.i, %mul4.i
  %4 = bitcast float %sub.i to i32
  %add.i = fadd float %mul6.i, %mul5.i
  %5 = bitcast float %add.i to i32
  %6 = tail call float @llvm.nvvm.fabs.f(float %sub.i) #2
  %7 = fcmp ugt float %6, 0x7FF0000000000000
  br i1 %7, label %land.lhs.true.i, label %_ZN12cuda_builtinmlIfEENS_7complexIT_EERKS3_S5_.exit

land.lhs.true.i:                                  ; preds = %entry
  %8 = tail call float @llvm.nvvm.fabs.f(float %add.i) #2
  %9 = fcmp ugt float %8, 0x7FF0000000000000
  br i1 %9, label %if.then.i, label %_ZN12cuda_builtinmlIfEENS_7complexIT_EERKS3_S5_.exit

if.then.i:                                        ; preds = %land.lhs.true.i
  %10 = tail call float @llvm.nvvm.fabs.f(float %0) #2
  %11 = fcmp oeq float %10, 0x7FF0000000000000
  %.pre.i = tail call float @llvm.nvvm.fabs.f(float %1) #2
  %12 = fcmp oeq float %.pre.i, 0x7FF0000000000000
  %or.cond.i = or i1 %11, %12
  br i1 %or.cond.i, label %if.then.14.i, label %if.end.31.i

if.then.14.i:                                     ; preds = %if.then.i
  %13 = bitcast float %0 to i32
  %14 = and i32 %13, -2147483648
  %15 = select i1 %11, i32 1065353216, i32 0
  %16 = or i32 %15, %14
  %17 = bitcast i32 %16 to float
  %18 = bitcast float %1 to i32
  %19 = and i32 %18, -2147483648
  %20 = select i1 %12, i32 1065353216, i32 0
  %21 = or i32 %20, %19
  %22 = bitcast i32 %21 to float
  %23 = tail call float @llvm.nvvm.fabs.f(float %2) #2
  %24 = fcmp ugt float %23, 0x7FF0000000000000
  br i1 %24, label %if.then.24.i, label %if.end.i

if.then.24.i:                                     ; preds = %if.then.14.i
  %25 = bitcast float %2 to i32
  %26 = and i32 %25, -2147483648
  %27 = bitcast i32 %26 to float
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.24.i, %if.then.14.i
  %__c.0.i = phi float [ %27, %if.then.24.i ], [ %2, %if.then.14.i ]
  %28 = tail call float @llvm.nvvm.fabs.f(float %3) #2
  %29 = fcmp ugt float %28, 0x7FF0000000000000
  br i1 %29, label %if.then.28.i, label %if.end.31.i

if.then.28.i:                                     ; preds = %if.end.i
  %30 = bitcast float %3 to i32
  %31 = and i32 %30, -2147483648
  %32 = bitcast i32 %31 to float
  br label %if.end.31.i

if.end.31.i:                                      ; preds = %if.then.28.i, %if.end.i, %if.then.i
  %__d.1.i = phi float [ %32, %if.then.28.i ], [ %3, %if.end.i ], [ %3, %if.then.i ]
  %__c.1.i = phi float [ %__c.0.i, %if.then.28.i ], [ %__c.0.i, %if.end.i ], [ %2, %if.then.i ]
  %__b.0.i = phi float [ %22, %if.then.28.i ], [ %22, %if.end.i ], [ %1, %if.then.i ]
  %__a.0.i = phi float [ %17, %if.then.28.i ], [ %17, %if.end.i ], [ %0, %if.then.i ]
  %__recalc.0.off0.i = phi i1 [ true, %if.then.28.i ], [ true, %if.end.i ], [ false, %if.then.i ]
  %33 = tail call float @llvm.nvvm.fabs.f(float %__c.1.i) #2
  %34 = fcmp oeq float %33, 0x7FF0000000000000
  %.pre6.i = tail call float @llvm.nvvm.fabs.f(float %__d.1.i) #2
  %35 = fcmp oeq float %.pre6.i, 0x7FF0000000000000
  %or.cond8.i = or i1 %34, %35
  br i1 %or.cond8.i, label %if.then.37.i, label %if.end.56.i

if.then.37.i:                                     ; preds = %if.end.31.i
  %36 = bitcast float %__c.1.i to i32
  %37 = and i32 %36, -2147483648
  %38 = select i1 %34, i32 1065353216, i32 0
  %39 = or i32 %38, %37
  %40 = bitcast i32 %39 to float
  %41 = bitcast float %__d.1.i to i32
  %42 = and i32 %41, -2147483648
  %43 = select i1 %35, i32 1065353216, i32 0
  %44 = or i32 %43, %42
  %45 = bitcast i32 %44 to float
  %46 = tail call float @llvm.nvvm.fabs.f(float %__a.0.i) #2
  %47 = fcmp ugt float %46, 0x7FF0000000000000
  br i1 %47, label %if.then.48.i, label %if.end.50.i

if.then.48.i:                                     ; preds = %if.then.37.i
  %48 = bitcast float %__a.0.i to i32
  %49 = and i32 %48, -2147483648
  %50 = bitcast i32 %49 to float
  br label %if.end.50.i

if.end.50.i:                                      ; preds = %if.then.48.i, %if.then.37.i
  %__a.1.i = phi float [ %50, %if.then.48.i ], [ %__a.0.i, %if.then.37.i ]
  %51 = tail call float @llvm.nvvm.fabs.f(float %__b.0.i) #2
  %52 = fcmp ugt float %51, 0x7FF0000000000000
  br i1 %52, label %if.then.53.i, label %if.then.93.i

if.then.53.i:                                     ; preds = %if.end.50.i
  %53 = bitcast float %__b.0.i to i32
  %54 = and i32 %53, -2147483648
  %55 = bitcast i32 %54 to float
  br label %if.then.93.i

if.end.56.i:                                      ; preds = %if.end.31.i
  br i1 %__recalc.0.off0.i, label %if.then.93.i, label %land.lhs.true.58.i

land.lhs.true.58.i:                               ; preds = %if.end.56.i
  %56 = tail call float @llvm.nvvm.fabs.f(float %mul.i) #2
  %57 = fcmp oeq float %56, 0x7FF0000000000000
  br i1 %57, label %if.then.70.i, label %lor.lhs.false.61.i

lor.lhs.false.61.i:                               ; preds = %land.lhs.true.58.i
  %58 = tail call float @llvm.nvvm.fabs.f(float %mul4.i) #2
  %59 = fcmp oeq float %58, 0x7FF0000000000000
  br i1 %59, label %if.then.70.i, label %lor.lhs.false.64.i

lor.lhs.false.64.i:                               ; preds = %lor.lhs.false.61.i
  %60 = tail call float @llvm.nvvm.fabs.f(float %mul5.i) #2
  %61 = fcmp oeq float %60, 0x7FF0000000000000
  br i1 %61, label %if.then.70.i, label %lor.lhs.false.67.i

lor.lhs.false.67.i:                               ; preds = %lor.lhs.false.64.i
  %62 = tail call float @llvm.nvvm.fabs.f(float %mul6.i) #2
  %63 = fcmp oeq float %62, 0x7FF0000000000000
  br i1 %63, label %if.then.70.i, label %_ZN12cuda_builtinmlIfEENS_7complexIT_EERKS3_S5_.exit

if.then.70.i:                                     ; preds = %lor.lhs.false.67.i, %lor.lhs.false.64.i, %lor.lhs.false.61.i, %land.lhs.true.58.i
  %64 = tail call float @llvm.nvvm.fabs.f(float %__a.0.i) #2
  %65 = fcmp ugt float %64, 0x7FF0000000000000
  br i1 %65, label %if.then.73.i, label %if.end.75.i

if.then.73.i:                                     ; preds = %if.then.70.i
  %66 = bitcast float %__a.0.i to i32
  %67 = and i32 %66, -2147483648
  %68 = bitcast i32 %67 to float
  br label %if.end.75.i

if.end.75.i:                                      ; preds = %if.then.73.i, %if.then.70.i
  %__a.3.i = phi float [ %68, %if.then.73.i ], [ %__a.0.i, %if.then.70.i ]
  %69 = tail call float @llvm.nvvm.fabs.f(float %__b.0.i) #2
  %70 = fcmp ugt float %69, 0x7FF0000000000000
  br i1 %70, label %if.then.78.i, label %if.end.80.i

if.then.78.i:                                     ; preds = %if.end.75.i
  %71 = bitcast float %__b.0.i to i32
  %72 = and i32 %71, -2147483648
  %73 = bitcast i32 %72 to float
  br label %if.end.80.i

if.end.80.i:                                      ; preds = %if.then.78.i, %if.end.75.i
  %__b.3.i = phi float [ %73, %if.then.78.i ], [ %__b.0.i, %if.end.75.i ]
  %74 = fcmp ugt float %33, 0x7FF0000000000000
  br i1 %74, label %if.then.83.i, label %if.end.85.i

if.then.83.i:                                     ; preds = %if.end.80.i
  %75 = bitcast float %__c.1.i to i32
  %76 = and i32 %75, -2147483648
  %77 = bitcast i32 %76 to float
  br label %if.end.85.i

if.end.85.i:                                      ; preds = %if.then.83.i, %if.end.80.i
  %__c.3.i = phi float [ %77, %if.then.83.i ], [ %__c.1.i, %if.end.80.i ]
  %78 = fcmp ugt float %.pre6.i, 0x7FF0000000000000
  br i1 %78, label %if.then.88.i, label %if.then.93.i

if.then.88.i:                                     ; preds = %if.end.85.i
  %79 = bitcast float %__d.1.i to i32
  %80 = and i32 %79, -2147483648
  %81 = bitcast i32 %80 to float
  br label %if.then.93.i

if.then.93.i:                                     ; preds = %if.then.88.i, %if.end.85.i, %if.end.56.i, %if.then.53.i, %if.end.50.i
  %__d.4.ph.i = phi float [ %__d.1.i, %if.end.85.i ], [ %81, %if.then.88.i ], [ %__d.1.i, %if.end.56.i ], [ %45, %if.end.50.i ], [ %45, %if.then.53.i ]
  %__c.4.ph.i = phi float [ %__c.3.i, %if.end.85.i ], [ %__c.3.i, %if.then.88.i ], [ %__c.1.i, %if.end.56.i ], [ %40, %if.end.50.i ], [ %40, %if.then.53.i ]
  %__b.4.ph.i = phi float [ %__b.3.i, %if.end.85.i ], [ %__b.3.i, %if.then.88.i ], [ %__b.0.i, %if.end.56.i ], [ %__b.0.i, %if.end.50.i ], [ %55, %if.then.53.i ]
  %__a.4.ph.i = phi float [ %__a.3.i, %if.end.85.i ], [ %__a.3.i, %if.then.88.i ], [ %__a.0.i, %if.end.56.i ], [ %__a.1.i, %if.end.50.i ], [ %__a.1.i, %if.then.53.i ]
  %mul95.i = fmul float %__c.4.ph.i, %__a.4.ph.i
  %mul96.i = fmul float %__d.4.ph.i, %__b.4.ph.i
  %sub97.i = fsub float %mul95.i, %mul96.i
  %mul98.i = fmul float %sub97.i, 0x7FF0000000000000
  %82 = bitcast float %mul98.i to i32
  %mul100.i = fmul float %__d.4.ph.i, %__a.4.ph.i
  %mul101.i = fmul float %__c.4.ph.i, %__b.4.ph.i
  %add102.i = fadd float %mul101.i, %mul100.i
  %mul103.i = fmul float %add102.i, 0x7FF0000000000000
  %83 = bitcast float %mul103.i to i32
  br label %_ZN12cuda_builtinmlIfEENS_7complexIT_EERKS3_S5_.exit

_ZN12cuda_builtinmlIfEENS_7complexIT_EERKS3_S5_.exit: ; preds = %if.then.93.i, %lor.lhs.false.67.i, %land.lhs.true.i, %entry
  %84 = phi i32 [ %4, %land.lhs.true.i ], [ %4, %entry ], [ %82, %if.then.93.i ], [ %4, %lor.lhs.false.67.i ]
  %85 = phi i32 [ %5, %land.lhs.true.i ], [ %5, %entry ], [ %83, %if.then.93.i ], [ %5, %lor.lhs.false.67.i ]
  %arrayidx5 = getelementptr inbounds float, ptr %f, i64 5
  store i32 %84, ptr %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds float, ptr %f, i64 6
  store i32 %85, ptr %arrayidx7, align 4
  ret void
}

declare float @llvm.nvvm.fabs.f(float)
