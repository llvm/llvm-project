; RUN: opt < %s -passes=loop-reroll -verify-scev -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void goo32(float alpha, float *a, float *b) {
;   for (int i = 0; i < 3200; i += 32) {
;     a[i] += alpha * b[i];
;     a[i + 1] += alpha * b[i + 1];
;     a[i + 2] += alpha * b[i + 2];
;     a[i + 3] += alpha * b[i + 3];
;     a[i + 4] += alpha * b[i + 4];
;     a[i + 5] += alpha * b[i + 5];
;     a[i + 6] += alpha * b[i + 6];
;     a[i + 7] += alpha * b[i + 7];
;     a[i + 8] += alpha * b[i + 8];
;     a[i + 9] += alpha * b[i + 9];
;     a[i + 10] += alpha * b[i + 10];
;     a[i + 11] += alpha * b[i + 11];
;     a[i + 12] += alpha * b[i + 12];
;     a[i + 13] += alpha * b[i + 13];
;     a[i + 14] += alpha * b[i + 14];
;     a[i + 15] += alpha * b[i + 15];
;     a[i + 16] += alpha * b[i + 16];
;     a[i + 17] += alpha * b[i + 17];
;     a[i + 18] += alpha * b[i + 18];
;     a[i + 19] += alpha * b[i + 19];
;     a[i + 20] += alpha * b[i + 20];
;     a[i + 21] += alpha * b[i + 21];
;     a[i + 22] += alpha * b[i + 22];
;     a[i + 23] += alpha * b[i + 23];
;     a[i + 24] += alpha * b[i + 24];
;     a[i + 25] += alpha * b[i + 25];
;     a[i + 26] += alpha * b[i + 26];
;     a[i + 27] += alpha * b[i + 27];
;     a[i + 28] += alpha * b[i + 28];
;     a[i + 29] += alpha * b[i + 29];
;     a[i + 30] += alpha * b[i + 30];
;     a[i + 31] += alpha * b[i + 31];
;   }
; }

; Function Attrs: norecurse nounwind uwtable
define void @goo32(float %alpha, ptr %a, ptr readonly %b) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %mul = fmul float %0, %alpha
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %1 = load float, ptr %arrayidx2, align 4
  %add = fadd float %1, %mul
  store float %add, ptr %arrayidx2, align 4
  %2 = or i64 %indvars.iv, 1
  %arrayidx5 = getelementptr inbounds float, ptr %b, i64 %2
  %3 = load float, ptr %arrayidx5, align 4
  %mul6 = fmul float %3, %alpha
  %arrayidx9 = getelementptr inbounds float, ptr %a, i64 %2
  %4 = load float, ptr %arrayidx9, align 4
  %add10 = fadd float %4, %mul6
  store float %add10, ptr %arrayidx9, align 4
  %5 = or i64 %indvars.iv, 2
  %arrayidx13 = getelementptr inbounds float, ptr %b, i64 %5
  %6 = load float, ptr %arrayidx13, align 4
  %mul14 = fmul float %6, %alpha
  %arrayidx17 = getelementptr inbounds float, ptr %a, i64 %5
  %7 = load float, ptr %arrayidx17, align 4
  %add18 = fadd float %7, %mul14
  store float %add18, ptr %arrayidx17, align 4
  %8 = or i64 %indvars.iv, 3
  %arrayidx21 = getelementptr inbounds float, ptr %b, i64 %8
  %9 = load float, ptr %arrayidx21, align 4
  %mul22 = fmul float %9, %alpha
  %arrayidx25 = getelementptr inbounds float, ptr %a, i64 %8
  %10 = load float, ptr %arrayidx25, align 4
  %add26 = fadd float %10, %mul22
  store float %add26, ptr %arrayidx25, align 4
  %11 = or i64 %indvars.iv, 4
  %arrayidx29 = getelementptr inbounds float, ptr %b, i64 %11
  %12 = load float, ptr %arrayidx29, align 4
  %mul30 = fmul float %12, %alpha
  %arrayidx33 = getelementptr inbounds float, ptr %a, i64 %11
  %13 = load float, ptr %arrayidx33, align 4
  %add34 = fadd float %13, %mul30
  store float %add34, ptr %arrayidx33, align 4
  %14 = or i64 %indvars.iv, 5
  %arrayidx37 = getelementptr inbounds float, ptr %b, i64 %14
  %15 = load float, ptr %arrayidx37, align 4
  %mul38 = fmul float %15, %alpha
  %arrayidx41 = getelementptr inbounds float, ptr %a, i64 %14
  %16 = load float, ptr %arrayidx41, align 4
  %add42 = fadd float %16, %mul38
  store float %add42, ptr %arrayidx41, align 4
  %17 = or i64 %indvars.iv, 6
  %arrayidx45 = getelementptr inbounds float, ptr %b, i64 %17
  %18 = load float, ptr %arrayidx45, align 4
  %mul46 = fmul float %18, %alpha
  %arrayidx49 = getelementptr inbounds float, ptr %a, i64 %17
  %19 = load float, ptr %arrayidx49, align 4
  %add50 = fadd float %19, %mul46
  store float %add50, ptr %arrayidx49, align 4
  %20 = or i64 %indvars.iv, 7
  %arrayidx53 = getelementptr inbounds float, ptr %b, i64 %20
  %21 = load float, ptr %arrayidx53, align 4
  %mul54 = fmul float %21, %alpha
  %arrayidx57 = getelementptr inbounds float, ptr %a, i64 %20
  %22 = load float, ptr %arrayidx57, align 4
  %add58 = fadd float %22, %mul54
  store float %add58, ptr %arrayidx57, align 4
  %23 = or i64 %indvars.iv, 8
  %arrayidx61 = getelementptr inbounds float, ptr %b, i64 %23
  %24 = load float, ptr %arrayidx61, align 4
  %mul62 = fmul float %24, %alpha
  %arrayidx65 = getelementptr inbounds float, ptr %a, i64 %23
  %25 = load float, ptr %arrayidx65, align 4
  %add66 = fadd float %25, %mul62
  store float %add66, ptr %arrayidx65, align 4
  %26 = or i64 %indvars.iv, 9
  %arrayidx69 = getelementptr inbounds float, ptr %b, i64 %26
  %27 = load float, ptr %arrayidx69, align 4
  %mul70 = fmul float %27, %alpha
  %arrayidx73 = getelementptr inbounds float, ptr %a, i64 %26
  %28 = load float, ptr %arrayidx73, align 4
  %add74 = fadd float %28, %mul70
  store float %add74, ptr %arrayidx73, align 4
  %29 = or i64 %indvars.iv, 10
  %arrayidx77 = getelementptr inbounds float, ptr %b, i64 %29
  %30 = load float, ptr %arrayidx77, align 4
  %mul78 = fmul float %30, %alpha
  %arrayidx81 = getelementptr inbounds float, ptr %a, i64 %29
  %31 = load float, ptr %arrayidx81, align 4
  %add82 = fadd float %31, %mul78
  store float %add82, ptr %arrayidx81, align 4
  %32 = or i64 %indvars.iv, 11
  %arrayidx85 = getelementptr inbounds float, ptr %b, i64 %32
  %33 = load float, ptr %arrayidx85, align 4
  %mul86 = fmul float %33, %alpha
  %arrayidx89 = getelementptr inbounds float, ptr %a, i64 %32
  %34 = load float, ptr %arrayidx89, align 4
  %add90 = fadd float %34, %mul86
  store float %add90, ptr %arrayidx89, align 4
  %35 = or i64 %indvars.iv, 12
  %arrayidx93 = getelementptr inbounds float, ptr %b, i64 %35
  %36 = load float, ptr %arrayidx93, align 4
  %mul94 = fmul float %36, %alpha
  %arrayidx97 = getelementptr inbounds float, ptr %a, i64 %35
  %37 = load float, ptr %arrayidx97, align 4
  %add98 = fadd float %37, %mul94
  store float %add98, ptr %arrayidx97, align 4
  %38 = or i64 %indvars.iv, 13
  %arrayidx101 = getelementptr inbounds float, ptr %b, i64 %38
  %39 = load float, ptr %arrayidx101, align 4
  %mul102 = fmul float %39, %alpha
  %arrayidx105 = getelementptr inbounds float, ptr %a, i64 %38
  %40 = load float, ptr %arrayidx105, align 4
  %add106 = fadd float %40, %mul102
  store float %add106, ptr %arrayidx105, align 4
  %41 = or i64 %indvars.iv, 14
  %arrayidx109 = getelementptr inbounds float, ptr %b, i64 %41
  %42 = load float, ptr %arrayidx109, align 4
  %mul110 = fmul float %42, %alpha
  %arrayidx113 = getelementptr inbounds float, ptr %a, i64 %41
  %43 = load float, ptr %arrayidx113, align 4
  %add114 = fadd float %43, %mul110
  store float %add114, ptr %arrayidx113, align 4
  %44 = or i64 %indvars.iv, 15
  %arrayidx117 = getelementptr inbounds float, ptr %b, i64 %44
  %45 = load float, ptr %arrayidx117, align 4
  %mul118 = fmul float %45, %alpha
  %arrayidx121 = getelementptr inbounds float, ptr %a, i64 %44
  %46 = load float, ptr %arrayidx121, align 4
  %add122 = fadd float %46, %mul118
  store float %add122, ptr %arrayidx121, align 4
  %47 = or i64 %indvars.iv, 16
  %arrayidx125 = getelementptr inbounds float, ptr %b, i64 %47
  %48 = load float, ptr %arrayidx125, align 4
  %mul126 = fmul float %48, %alpha
  %arrayidx129 = getelementptr inbounds float, ptr %a, i64 %47
  %49 = load float, ptr %arrayidx129, align 4
  %add130 = fadd float %49, %mul126
  store float %add130, ptr %arrayidx129, align 4
  %50 = or i64 %indvars.iv, 17
  %arrayidx133 = getelementptr inbounds float, ptr %b, i64 %50
  %51 = load float, ptr %arrayidx133, align 4
  %mul134 = fmul float %51, %alpha
  %arrayidx137 = getelementptr inbounds float, ptr %a, i64 %50
  %52 = load float, ptr %arrayidx137, align 4
  %add138 = fadd float %52, %mul134
  store float %add138, ptr %arrayidx137, align 4
  %53 = or i64 %indvars.iv, 18
  %arrayidx141 = getelementptr inbounds float, ptr %b, i64 %53
  %54 = load float, ptr %arrayidx141, align 4
  %mul142 = fmul float %54, %alpha
  %arrayidx145 = getelementptr inbounds float, ptr %a, i64 %53
  %55 = load float, ptr %arrayidx145, align 4
  %add146 = fadd float %55, %mul142
  store float %add146, ptr %arrayidx145, align 4
  %56 = or i64 %indvars.iv, 19
  %arrayidx149 = getelementptr inbounds float, ptr %b, i64 %56
  %57 = load float, ptr %arrayidx149, align 4
  %mul150 = fmul float %57, %alpha
  %arrayidx153 = getelementptr inbounds float, ptr %a, i64 %56
  %58 = load float, ptr %arrayidx153, align 4
  %add154 = fadd float %58, %mul150
  store float %add154, ptr %arrayidx153, align 4
  %59 = or i64 %indvars.iv, 20
  %arrayidx157 = getelementptr inbounds float, ptr %b, i64 %59
  %60 = load float, ptr %arrayidx157, align 4
  %mul158 = fmul float %60, %alpha
  %arrayidx161 = getelementptr inbounds float, ptr %a, i64 %59
  %61 = load float, ptr %arrayidx161, align 4
  %add162 = fadd float %61, %mul158
  store float %add162, ptr %arrayidx161, align 4
  %62 = or i64 %indvars.iv, 21
  %arrayidx165 = getelementptr inbounds float, ptr %b, i64 %62
  %63 = load float, ptr %arrayidx165, align 4
  %mul166 = fmul float %63, %alpha
  %arrayidx169 = getelementptr inbounds float, ptr %a, i64 %62
  %64 = load float, ptr %arrayidx169, align 4
  %add170 = fadd float %64, %mul166
  store float %add170, ptr %arrayidx169, align 4
  %65 = or i64 %indvars.iv, 22
  %arrayidx173 = getelementptr inbounds float, ptr %b, i64 %65
  %66 = load float, ptr %arrayidx173, align 4
  %mul174 = fmul float %66, %alpha
  %arrayidx177 = getelementptr inbounds float, ptr %a, i64 %65
  %67 = load float, ptr %arrayidx177, align 4
  %add178 = fadd float %67, %mul174
  store float %add178, ptr %arrayidx177, align 4
  %68 = or i64 %indvars.iv, 23
  %arrayidx181 = getelementptr inbounds float, ptr %b, i64 %68
  %69 = load float, ptr %arrayidx181, align 4
  %mul182 = fmul float %69, %alpha
  %arrayidx185 = getelementptr inbounds float, ptr %a, i64 %68
  %70 = load float, ptr %arrayidx185, align 4
  %add186 = fadd float %70, %mul182
  store float %add186, ptr %arrayidx185, align 4
  %71 = or i64 %indvars.iv, 24
  %arrayidx189 = getelementptr inbounds float, ptr %b, i64 %71
  %72 = load float, ptr %arrayidx189, align 4
  %mul190 = fmul float %72, %alpha
  %arrayidx193 = getelementptr inbounds float, ptr %a, i64 %71
  %73 = load float, ptr %arrayidx193, align 4
  %add194 = fadd float %73, %mul190
  store float %add194, ptr %arrayidx193, align 4
  %74 = or i64 %indvars.iv, 25
  %arrayidx197 = getelementptr inbounds float, ptr %b, i64 %74
  %75 = load float, ptr %arrayidx197, align 4
  %mul198 = fmul float %75, %alpha
  %arrayidx201 = getelementptr inbounds float, ptr %a, i64 %74
  %76 = load float, ptr %arrayidx201, align 4
  %add202 = fadd float %76, %mul198
  store float %add202, ptr %arrayidx201, align 4
  %77 = or i64 %indvars.iv, 26
  %arrayidx205 = getelementptr inbounds float, ptr %b, i64 %77
  %78 = load float, ptr %arrayidx205, align 4
  %mul206 = fmul float %78, %alpha
  %arrayidx209 = getelementptr inbounds float, ptr %a, i64 %77
  %79 = load float, ptr %arrayidx209, align 4
  %add210 = fadd float %79, %mul206
  store float %add210, ptr %arrayidx209, align 4
  %80 = or i64 %indvars.iv, 27
  %arrayidx213 = getelementptr inbounds float, ptr %b, i64 %80
  %81 = load float, ptr %arrayidx213, align 4
  %mul214 = fmul float %81, %alpha
  %arrayidx217 = getelementptr inbounds float, ptr %a, i64 %80
  %82 = load float, ptr %arrayidx217, align 4
  %add218 = fadd float %82, %mul214
  store float %add218, ptr %arrayidx217, align 4
  %83 = or i64 %indvars.iv, 28
  %arrayidx221 = getelementptr inbounds float, ptr %b, i64 %83
  %84 = load float, ptr %arrayidx221, align 4
  %mul222 = fmul float %84, %alpha
  %arrayidx225 = getelementptr inbounds float, ptr %a, i64 %83
  %85 = load float, ptr %arrayidx225, align 4
  %add226 = fadd float %85, %mul222
  store float %add226, ptr %arrayidx225, align 4
  %86 = or i64 %indvars.iv, 29
  %arrayidx229 = getelementptr inbounds float, ptr %b, i64 %86
  %87 = load float, ptr %arrayidx229, align 4
  %mul230 = fmul float %87, %alpha
  %arrayidx233 = getelementptr inbounds float, ptr %a, i64 %86
  %88 = load float, ptr %arrayidx233, align 4
  %add234 = fadd float %88, %mul230
  store float %add234, ptr %arrayidx233, align 4
  %89 = or i64 %indvars.iv, 30
  %arrayidx237 = getelementptr inbounds float, ptr %b, i64 %89
  %90 = load float, ptr %arrayidx237, align 4
  %mul238 = fmul float %90, %alpha
  %arrayidx241 = getelementptr inbounds float, ptr %a, i64 %89
  %91 = load float, ptr %arrayidx241, align 4
  %add242 = fadd float %91, %mul238
  store float %add242, ptr %arrayidx241, align 4
  %92 = or i64 %indvars.iv, 31
  %arrayidx245 = getelementptr inbounds float, ptr %b, i64 %92
  %93 = load float, ptr %arrayidx245, align 4
  %mul246 = fmul float %93, %alpha
  %arrayidx249 = getelementptr inbounds float, ptr %a, i64 %92
  %94 = load float, ptr %arrayidx249, align 4
  %add250 = fadd float %94, %mul246
  store float %add250, ptr %arrayidx249, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 32
  %cmp = icmp slt i64 %indvars.iv.next, 3200
  br i1 %cmp, label %for.body, label %for.end

; CHECK-LABEL: @goo32

; CHECK: for.body:
; CHECK: %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %entry ]
; CHECK: %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvar
; CHECK: %0 = load float, ptr %arrayidx, align 4
; CHECK: %mul = fmul float %0, %alpha
; CHECK: %arrayidx2 = getelementptr inbounds float, ptr %a, i64 %indvar
; CHECK: %1 = load float, ptr %arrayidx2, align 4
; CHECK: %add = fadd float %1, %mul
; CHECK: store float %add, ptr %arrayidx2, align 4
; CHECK: %indvar.next = add i64 %indvar, 1
; CHECK: %exitcond = icmp eq i64 %indvar, 3199
; CHECK: br i1 %exitcond, label %for.end, label %for.body
; CHECK: ret

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { nounwind uwtable }
