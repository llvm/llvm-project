; RUN: opt %loadPolly -polly-parallel -polly-vectorizer=stripmine -polly-codegen-verify -polly-opt-isl -polly-print-ast -polly-codegen -disable-output < %s | FileCheck %s
;
; Check that there are no nested #pragma omp parallel for inside a
; #pragma omp parallel for loop.
; See llvm.org/PR38073 and llvm.org/PR33153
;
; This test unfortunately is very dependent on the result of the schedule
; optimizer (-polly-opt-isl).
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@b = external dso_local unnamed_addr global [1984 x [1984 x double]], align 16
@c = external dso_local unnamed_addr global [1984 x [1984 x double]], align 16

define dso_local void @main() local_unnamed_addr {
entry:
  %cond = select i1 undef, i32 undef, i32 1984
  %tmp = zext i32 %cond to i64
  %cond63 = select i1 undef, i32 undef, i32 1984
  %tmp1 = zext i32 %cond63 to i64
  br label %for.cond51.preheader

for.cond51.preheader:
  %indvars.iv213 = phi i64 [ 0, %entry ], [ %indvars.iv.next214, %for.inc98 ]
  %cond73 = select i1 undef, i32 undef, i32 1984
  %tmp2 = zext i32 %cond73 to i64
  br label %for.cond56.preheader

for.cond56.preheader:
  %indvars.iv223 = phi i64 [ 0, %for.cond51.preheader ], [ %indvars.iv.next224, %for.inc95 ]
  br label %for.cond66.preheader

for.cond66.preheader:
  %indvars.iv219 = phi i64 [ %indvars.iv.next220, %for.inc92 ], [ 0, %for.cond56.preheader ]
  br label %for.body75

for.body75:
  %indvars.iv215 = phi i64 [ %indvars.iv213, %for.cond66.preheader ], [ %indvars.iv.next216, %for.body75 ]
  %arrayidx83 = getelementptr inbounds [1984 x [1984 x double]], [1984 x [1984 x double]]* @b, i64 0, i64 %indvars.iv219, i64 %indvars.iv215
  %tmp3 = load double, double* %arrayidx83, align 8
  %arrayidx87 = getelementptr inbounds [1984 x [1984 x double]], [1984 x [1984 x double]]* @c, i64 0, i64 %indvars.iv223, i64 %indvars.iv215
  store double undef, double* %arrayidx87, align 8
  %indvars.iv.next216 = add nuw nsw i64 %indvars.iv215, 1
  %cmp74 = icmp ult i64 %indvars.iv.next216, %tmp2
  br i1 %cmp74, label %for.body75, label %for.inc92

for.inc92:
  %indvars.iv.next220 = add nuw nsw i64 %indvars.iv219, 1
  %cmp64 = icmp ult i64 %indvars.iv.next220, %tmp1
  br i1 %cmp64, label %for.cond66.preheader, label %for.inc95

for.inc95:
  %indvars.iv.next224 = add nuw nsw i64 %indvars.iv223, 1
  %cmp54 = icmp ult i64 %indvars.iv.next224, %tmp
  br i1 %cmp54, label %for.cond56.preheader, label %for.inc98

for.inc98:
  %indvars.iv.next214 = add nuw nsw i64 %indvars.iv213, 48
  br label %for.cond51.preheader
}

; No parallel loop except the to outermost.
; CHECK: #pragma omp parallel for
; CHECK: #pragma omp parallel for
; CHECK-NOT: #pragma omp parallel for
