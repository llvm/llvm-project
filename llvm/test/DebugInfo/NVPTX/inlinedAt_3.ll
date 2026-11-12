; RUN: llc < %s -mattr=+ptx72 | FileCheck %s
;
;; Test inlining of a C++ constructor with control flow - verifies that inlined_at
;; information is correctly emitted when a constructor containing loops and conditionals is inlined.
;
; __device__ int gg;
; __device__ int *arr;
;
; class C {
;   int priv;
;   public: __device__ C();
;   __device__ C(int);
;   __device__ int get() const;
; };
;
;
; __device__ C::C() : priv(1) {
;   int sum = 0;
;   for (int i = 0; i < gg; ++i) sum += arr[i];
;   if (sum > 17)
;     priv = sum;
; }
;
; __device__ C::C(int n) : priv(n) {}
;
; __device__ int C::get() const { return priv; }
;
; __global__ void kernel(int n) {
;   C c1;
;   if (n > 7)
;     gg = c1.get();
; }
;
; CHECK: .loc [[FILENUM:[1-9]]] 24
; CHECK: .loc [[FILENUM]] 14 {{[0-9]*}}, function_name [[CTORNAME:\$L__info_string[0-9]+]], inlined_at [[FILENUM]] 24
; CHECK: .section .debug_str
; CHECK: {
; CHECK: [[CTORNAME]]:
; CHECK-NEXT: // {{.*}} _ZN1CC1Ev
; CHECK: }

source_filename = "<unnamed>"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8-p6:32:32"
target triple = "nvptx64-nvidia-cuda"

@gg = internal addrspace(1) global i32 0, align 4
@arr = internal addrspace(1) global ptr null, align 8
@llvm.used = appending global [3 x ptr] [ptr @_Z6kerneli, ptr addrspacecast (ptr addrspace(1) @arr to ptr), ptr addrspacecast (ptr addrspace(1) @gg to ptr)], section "llvm.metadata"

define void @_Z6kerneli(i32 noundef %n) !dbg !4 {
entry:
  %tmp3.i1 = load i32, ptr addrspace(1) @gg, align 4, !dbg !6
  %cmp.i2 = icmp sgt i32 %tmp3.i1, 0, !dbg !6
  br i1 %cmp.i2, label %for.body.i.preheader, label %for.end.i, !dbg !6

for.body.i.preheader:                             ; preds = %entry
  %tmp4.i = load ptr, ptr addrspace(1) @arr, align 8
  %0 = addrspacecast ptr %tmp4.i to ptr addrspace(1)
  %xtraiter = and i32 %tmp3.i1, 15, !dbg !6
  %1 = icmp samesign ult i32 %tmp3.i1, 16, !dbg !6
  br i1 %1, label %for.end.i.loopexit.unr-lcssa, label %for.body.i.preheader.new, !dbg !6

for.body.i.preheader.new:                         ; preds = %for.body.i.preheader
  %unroll_iter = and i32 %tmp3.i1, 2147483632, !dbg !6
  br label %for.body.i, !dbg !6

for.body.i:                                       ; preds = %for.body.i, %for.body.i.preheader.new
  %i.0.i4 = phi i32 [ 0, %for.body.i.preheader.new ], [ %inc.i.15, %for.body.i ]
  %sum.0.i3 = phi i32 [ 0, %for.body.i.preheader.new ], [ %add.i.15, %for.body.i ]
  %niter = phi i32 [ 0, %for.body.i.preheader.new ], [ %niter.next.15, %for.body.i ]
  %2 = zext nneg i32 %i.0.i4 to i64, !dbg !6
  %getElem = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %2, !dbg !6
  %tmp6.i = load i32, ptr addrspace(1) %getElem, align 4, !dbg !6
  %add.i = add nsw i32 %tmp6.i, %sum.0.i3, !dbg !6
  %inc.i = add nuw nsw i32 %i.0.i4, 1, !dbg !6
  %3 = zext nneg i32 %inc.i to i64, !dbg !6
  %getElem.1 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %3, !dbg !6
  %tmp6.i.1 = load i32, ptr addrspace(1) %getElem.1, align 4, !dbg !6
  %add.i.1 = add nsw i32 %tmp6.i.1, %add.i, !dbg !6
  %inc.i.1 = add nuw nsw i32 %i.0.i4, 2, !dbg !6
  %4 = zext nneg i32 %inc.i.1 to i64, !dbg !6
  %getElem.2 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %4, !dbg !6
  %tmp6.i.2 = load i32, ptr addrspace(1) %getElem.2, align 4, !dbg !6
  %add.i.2 = add nsw i32 %tmp6.i.2, %add.i.1, !dbg !6
  %inc.i.2 = add nuw nsw i32 %i.0.i4, 3, !dbg !6
  %5 = zext nneg i32 %inc.i.2 to i64, !dbg !6
  %getElem.3 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %5, !dbg !6
  %tmp6.i.3 = load i32, ptr addrspace(1) %getElem.3, align 4, !dbg !6
  %add.i.3 = add nsw i32 %tmp6.i.3, %add.i.2, !dbg !6
  %inc.i.3 = add nuw nsw i32 %i.0.i4, 4, !dbg !6
  %6 = zext nneg i32 %inc.i.3 to i64, !dbg !6
  %getElem.4 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %6, !dbg !6
  %tmp6.i.4 = load i32, ptr addrspace(1) %getElem.4, align 4, !dbg !6
  %add.i.4 = add nsw i32 %tmp6.i.4, %add.i.3, !dbg !6
  %inc.i.4 = add nuw nsw i32 %i.0.i4, 5, !dbg !6
  %7 = zext nneg i32 %inc.i.4 to i64, !dbg !6
  %getElem.5 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %7, !dbg !6
  %tmp6.i.5 = load i32, ptr addrspace(1) %getElem.5, align 4, !dbg !6
  %add.i.5 = add nsw i32 %tmp6.i.5, %add.i.4, !dbg !6
  %inc.i.5 = add nuw nsw i32 %i.0.i4, 6, !dbg !6
  %8 = zext nneg i32 %inc.i.5 to i64, !dbg !6
  %getElem.6 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %8, !dbg !6
  %tmp6.i.6 = load i32, ptr addrspace(1) %getElem.6, align 4, !dbg !6
  %add.i.6 = add nsw i32 %tmp6.i.6, %add.i.5, !dbg !6
  %inc.i.6 = add nuw nsw i32 %i.0.i4, 7, !dbg !6
  %9 = zext nneg i32 %inc.i.6 to i64, !dbg !6
  %getElem.7 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %9, !dbg !6
  %tmp6.i.7 = load i32, ptr addrspace(1) %getElem.7, align 4, !dbg !6
  %add.i.7 = add nsw i32 %tmp6.i.7, %add.i.6, !dbg !6
  %inc.i.7 = add nuw nsw i32 %i.0.i4, 8, !dbg !6
  %10 = zext nneg i32 %inc.i.7 to i64, !dbg !6
  %getElem.8 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %10, !dbg !6
  %tmp6.i.8 = load i32, ptr addrspace(1) %getElem.8, align 4, !dbg !6
  %add.i.8 = add nsw i32 %tmp6.i.8, %add.i.7, !dbg !6
  %inc.i.8 = add nuw nsw i32 %i.0.i4, 9, !dbg !6
  %11 = zext nneg i32 %inc.i.8 to i64, !dbg !6
  %getElem.9 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %11, !dbg !6
  %tmp6.i.9 = load i32, ptr addrspace(1) %getElem.9, align 4, !dbg !6
  %add.i.9 = add nsw i32 %tmp6.i.9, %add.i.8, !dbg !6
  %inc.i.9 = add nuw nsw i32 %i.0.i4, 10, !dbg !6
  %12 = zext nneg i32 %inc.i.9 to i64, !dbg !6
  %getElem.10 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %12, !dbg !6
  %tmp6.i.10 = load i32, ptr addrspace(1) %getElem.10, align 4, !dbg !6
  %add.i.10 = add nsw i32 %tmp6.i.10, %add.i.9, !dbg !6
  %inc.i.10 = add nuw nsw i32 %i.0.i4, 11, !dbg !6
  %13 = zext nneg i32 %inc.i.10 to i64, !dbg !6
  %getElem.11 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %13, !dbg !6
  %tmp6.i.11 = load i32, ptr addrspace(1) %getElem.11, align 4, !dbg !6
  %add.i.11 = add nsw i32 %tmp6.i.11, %add.i.10, !dbg !6
  %inc.i.11 = add nuw nsw i32 %i.0.i4, 12, !dbg !6
  %14 = zext nneg i32 %inc.i.11 to i64, !dbg !6
  %getElem.12 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %14, !dbg !6
  %tmp6.i.12 = load i32, ptr addrspace(1) %getElem.12, align 4, !dbg !6
  %add.i.12 = add nsw i32 %tmp6.i.12, %add.i.11, !dbg !6
  %inc.i.12 = add nuw nsw i32 %i.0.i4, 13, !dbg !6
  %15 = zext nneg i32 %inc.i.12 to i64, !dbg !6
  %getElem.13 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %15, !dbg !6
  %tmp6.i.13 = load i32, ptr addrspace(1) %getElem.13, align 4, !dbg !6
  %add.i.13 = add nsw i32 %tmp6.i.13, %add.i.12, !dbg !6
  %inc.i.13 = add nuw nsw i32 %i.0.i4, 14, !dbg !6
  %16 = zext nneg i32 %inc.i.13 to i64, !dbg !6
  %getElem.14 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %16, !dbg !6
  %tmp6.i.14 = load i32, ptr addrspace(1) %getElem.14, align 4, !dbg !6
  %add.i.14 = add nsw i32 %tmp6.i.14, %add.i.13, !dbg !6
  %inc.i.14 = add nuw nsw i32 %i.0.i4, 15, !dbg !6
  %17 = zext nneg i32 %inc.i.14 to i64, !dbg !6
  %getElem.15 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %17, !dbg !6
  %tmp6.i.15 = load i32, ptr addrspace(1) %getElem.15, align 4, !dbg !6
  %add.i.15 = add nsw i32 %tmp6.i.15, %add.i.14, !dbg !6
  %inc.i.15 = add nuw nsw i32 %i.0.i4, 16, !dbg !6
  %niter.next.15 = add i32 %niter, 16, !dbg !6
  %niter.ncmp.15.not = icmp eq i32 %niter.next.15, %unroll_iter, !dbg !6
  br i1 %niter.ncmp.15.not, label %for.end.i.loopexit.unr-lcssa, label %for.body.i, !dbg !6, !llvm.loop !11

for.end.i.loopexit.unr-lcssa:                     ; preds = %for.body.i, %for.body.i.preheader
  %add.i.lcssa.ph = phi i32 [ poison, %for.body.i.preheader ], [ %add.i.15, %for.body.i ]
  %i.0.i4.unr = phi i32 [ 0, %for.body.i.preheader ], [ %inc.i.15, %for.body.i ]
  %sum.0.i3.unr = phi i32 [ 0, %for.body.i.preheader ], [ %add.i.15, %for.body.i ]
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0, !dbg !6
  br i1 %lcmp.mod.not, label %for.end.i, label %for.body.i.epil.preheader, !dbg !6

for.body.i.epil.preheader:                        ; preds = %for.end.i.loopexit.unr-lcssa
  %xtraiter6 = and i32 %tmp3.i1, 7, !dbg !6
  %18 = icmp samesign ult i32 %xtraiter, 8, !dbg !6
  br i1 %18, label %for.end.i.loopexit.epilog-lcssa.unr-lcssa, label %for.body.i.epil, !dbg !6

for.body.i.epil:                                  ; preds = %for.body.i.epil.preheader
  %19 = zext nneg i32 %i.0.i4.unr to i64, !dbg !6
  %getElem.epil = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %19, !dbg !6
  %tmp6.i.epil = load i32, ptr addrspace(1) %getElem.epil, align 4, !dbg !6
  %add.i.epil = add nsw i32 %tmp6.i.epil, %sum.0.i3.unr, !dbg !6
  %inc.i.epil = add nuw nsw i32 %i.0.i4.unr, 1, !dbg !6
  %20 = zext nneg i32 %inc.i.epil to i64, !dbg !6
  %getElem.epil.1 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %20, !dbg !6
  %tmp6.i.epil.1 = load i32, ptr addrspace(1) %getElem.epil.1, align 4, !dbg !6
  %add.i.epil.1 = add nsw i32 %tmp6.i.epil.1, %add.i.epil, !dbg !6
  %inc.i.epil.1 = add nuw nsw i32 %i.0.i4.unr, 2, !dbg !6
  %21 = zext nneg i32 %inc.i.epil.1 to i64, !dbg !6
  %getElem.epil.2 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %21, !dbg !6
  %tmp6.i.epil.2 = load i32, ptr addrspace(1) %getElem.epil.2, align 4, !dbg !6
  %add.i.epil.2 = add nsw i32 %tmp6.i.epil.2, %add.i.epil.1, !dbg !6
  %inc.i.epil.2 = add nuw nsw i32 %i.0.i4.unr, 3, !dbg !6
  %22 = zext nneg i32 %inc.i.epil.2 to i64, !dbg !6
  %getElem.epil.3 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %22, !dbg !6
  %tmp6.i.epil.3 = load i32, ptr addrspace(1) %getElem.epil.3, align 4, !dbg !6
  %add.i.epil.3 = add nsw i32 %tmp6.i.epil.3, %add.i.epil.2, !dbg !6
  %inc.i.epil.3 = add nuw nsw i32 %i.0.i4.unr, 4, !dbg !6
  %23 = zext nneg i32 %inc.i.epil.3 to i64, !dbg !6
  %getElem.epil.4 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %23, !dbg !6
  %tmp6.i.epil.4 = load i32, ptr addrspace(1) %getElem.epil.4, align 4, !dbg !6
  %add.i.epil.4 = add nsw i32 %tmp6.i.epil.4, %add.i.epil.3, !dbg !6
  %inc.i.epil.4 = add nuw nsw i32 %i.0.i4.unr, 5, !dbg !6
  %24 = zext nneg i32 %inc.i.epil.4 to i64, !dbg !6
  %getElem.epil.5 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %24, !dbg !6
  %tmp6.i.epil.5 = load i32, ptr addrspace(1) %getElem.epil.5, align 4, !dbg !6
  %add.i.epil.5 = add nsw i32 %tmp6.i.epil.5, %add.i.epil.4, !dbg !6
  %inc.i.epil.5 = add nuw nsw i32 %i.0.i4.unr, 6, !dbg !6
  %25 = zext nneg i32 %inc.i.epil.5 to i64, !dbg !6
  %getElem.epil.6 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %25, !dbg !6
  %tmp6.i.epil.6 = load i32, ptr addrspace(1) %getElem.epil.6, align 4, !dbg !6
  %add.i.epil.6 = add nsw i32 %tmp6.i.epil.6, %add.i.epil.5, !dbg !6
  %inc.i.epil.6 = add nuw nsw i32 %i.0.i4.unr, 7, !dbg !6
  %26 = zext nneg i32 %inc.i.epil.6 to i64, !dbg !6
  %getElem.epil.7 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %26, !dbg !6
  %tmp6.i.epil.7 = load i32, ptr addrspace(1) %getElem.epil.7, align 4, !dbg !6
  %add.i.epil.7 = add nsw i32 %tmp6.i.epil.7, %add.i.epil.6, !dbg !6
  %inc.i.epil.7 = add nuw nsw i32 %i.0.i4.unr, 8, !dbg !6
  br label %for.end.i.loopexit.epilog-lcssa.unr-lcssa, !dbg !6

for.end.i.loopexit.epilog-lcssa.unr-lcssa:        ; preds = %for.body.i.epil, %for.body.i.epil.preheader
  %add.i.lcssa.ph5.ph = phi i32 [ poison, %for.body.i.epil.preheader ], [ %add.i.epil.7, %for.body.i.epil ]
  %i.0.i4.epil.unr = phi i32 [ %i.0.i4.unr, %for.body.i.epil.preheader ], [ %inc.i.epil.7, %for.body.i.epil ]
  %sum.0.i3.epil.unr = phi i32 [ %sum.0.i3.unr, %for.body.i.epil.preheader ], [ %add.i.epil.7, %for.body.i.epil ]
  %lcmp.mod8.not = icmp eq i32 %xtraiter6, 0, !dbg !6
  br i1 %lcmp.mod8.not, label %for.end.i, label %for.body.i.epil.epil.preheader, !dbg !6

for.body.i.epil.epil.preheader:                   ; preds = %for.end.i.loopexit.epilog-lcssa.unr-lcssa
  %xtraiter12 = and i32 %tmp3.i1, 3, !dbg !6
  %27 = icmp samesign ult i32 %xtraiter6, 4, !dbg !6
  br i1 %27, label %for.end.i.loopexit.epilog-lcssa.epilog-lcssa.unr-lcssa, label %for.body.i.epil.epil, !dbg !6

for.body.i.epil.epil:                             ; preds = %for.body.i.epil.epil.preheader
  %28 = zext nneg i32 %i.0.i4.epil.unr to i64, !dbg !6
  %getElem.epil.epil = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %28, !dbg !6
  %tmp6.i.epil.epil = load i32, ptr addrspace(1) %getElem.epil.epil, align 4, !dbg !6
  %add.i.epil.epil = add nsw i32 %tmp6.i.epil.epil, %sum.0.i3.epil.unr, !dbg !6
  %inc.i.epil.epil = add nuw nsw i32 %i.0.i4.epil.unr, 1, !dbg !6
  %29 = zext nneg i32 %inc.i.epil.epil to i64, !dbg !6
  %getElem.epil.epil.1 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %29, !dbg !6
  %tmp6.i.epil.epil.1 = load i32, ptr addrspace(1) %getElem.epil.epil.1, align 4, !dbg !6
  %add.i.epil.epil.1 = add nsw i32 %tmp6.i.epil.epil.1, %add.i.epil.epil, !dbg !6
  %inc.i.epil.epil.1 = add nuw nsw i32 %i.0.i4.epil.unr, 2, !dbg !6
  %30 = zext nneg i32 %inc.i.epil.epil.1 to i64, !dbg !6
  %getElem.epil.epil.2 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %30, !dbg !6
  %tmp6.i.epil.epil.2 = load i32, ptr addrspace(1) %getElem.epil.epil.2, align 4, !dbg !6
  %add.i.epil.epil.2 = add nsw i32 %tmp6.i.epil.epil.2, %add.i.epil.epil.1, !dbg !6
  %inc.i.epil.epil.2 = add nuw nsw i32 %i.0.i4.epil.unr, 3, !dbg !6
  %31 = zext nneg i32 %inc.i.epil.epil.2 to i64, !dbg !6
  %getElem.epil.epil.3 = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %31, !dbg !6
  %tmp6.i.epil.epil.3 = load i32, ptr addrspace(1) %getElem.epil.epil.3, align 4, !dbg !6
  %add.i.epil.epil.3 = add nsw i32 %tmp6.i.epil.epil.3, %add.i.epil.epil.2, !dbg !6
  %inc.i.epil.epil.3 = add nuw nsw i32 %i.0.i4.epil.unr, 4, !dbg !6
  br label %for.end.i.loopexit.epilog-lcssa.epilog-lcssa.unr-lcssa, !dbg !6

for.end.i.loopexit.epilog-lcssa.epilog-lcssa.unr-lcssa: ; preds = %for.body.i.epil.epil, %for.body.i.epil.epil.preheader
  %add.i.lcssa.ph5.ph9.ph = phi i32 [ poison, %for.body.i.epil.epil.preheader ], [ %add.i.epil.epil.3, %for.body.i.epil.epil ]
  %i.0.i4.epil.epil.unr = phi i32 [ %i.0.i4.epil.unr, %for.body.i.epil.epil.preheader ], [ %inc.i.epil.epil.3, %for.body.i.epil.epil ]
  %sum.0.i3.epil.epil.unr = phi i32 [ %sum.0.i3.epil.unr, %for.body.i.epil.epil.preheader ], [ %add.i.epil.epil.3, %for.body.i.epil.epil ]
  %lcmp.mod14.not = icmp eq i32 %xtraiter12, 0, !dbg !6
  br i1 %lcmp.mod14.not, label %for.end.i, label %for.body.i.epil.epil.epil, !dbg !6

for.body.i.epil.epil.epil:                        ; preds = %for.end.i.loopexit.epilog-lcssa.epilog-lcssa.unr-lcssa, %for.body.i.epil.epil.epil
  %i.0.i4.epil.epil.epil = phi i32 [ %inc.i.epil.epil.epil, %for.body.i.epil.epil.epil ], [ %i.0.i4.epil.epil.unr, %for.end.i.loopexit.epilog-lcssa.epilog-lcssa.unr-lcssa ]
  %sum.0.i3.epil.epil.epil = phi i32 [ %add.i.epil.epil.epil, %for.body.i.epil.epil.epil ], [ %sum.0.i3.epil.epil.unr, %for.end.i.loopexit.epilog-lcssa.epilog-lcssa.unr-lcssa ]
  %epil.iter13 = phi i32 [ %epil.iter13.next, %for.body.i.epil.epil.epil ], [ 0, %for.end.i.loopexit.epilog-lcssa.epilog-lcssa.unr-lcssa ]
  %32 = zext nneg i32 %i.0.i4.epil.epil.epil to i64, !dbg !6
  %getElem.epil.epil.epil = getelementptr inbounds nuw i32, ptr addrspace(1) %0, i64 %32, !dbg !6
  %tmp6.i.epil.epil.epil = load i32, ptr addrspace(1) %getElem.epil.epil.epil, align 4, !dbg !6
  %add.i.epil.epil.epil = add nsw i32 %tmp6.i.epil.epil.epil, %sum.0.i3.epil.epil.epil, !dbg !6
  %inc.i.epil.epil.epil = add nuw nsw i32 %i.0.i4.epil.epil.epil, 1, !dbg !6
  %epil.iter13.next = add i32 %epil.iter13, 1, !dbg !6
  %epil.iter13.cmp.not = icmp eq i32 %epil.iter13.next, %xtraiter12, !dbg !6
  br i1 %epil.iter13.cmp.not, label %for.end.i, label %for.body.i.epil.epil.epil, !dbg !6, !llvm.loop !14

for.end.i:                                        ; preds = %for.body.i.epil.epil.epil, %for.end.i.loopexit.unr-lcssa, %for.end.i.loopexit.epilog-lcssa.epilog-lcssa.unr-lcssa, %for.end.i.loopexit.epilog-lcssa.unr-lcssa, %entry
  %sum.0.i.lcssa = phi i32 [ 0, %entry ], [ %add.i.lcssa.ph, %for.end.i.loopexit.unr-lcssa ], [ %add.i.lcssa.ph5.ph, %for.end.i.loopexit.epilog-lcssa.unr-lcssa ], [ %add.i.lcssa.ph5.ph9.ph, %for.end.i.loopexit.epilog-lcssa.epilog-lcssa.unr-lcssa ], [ %add.i.epil.epil.epil, %for.body.i.epil.epil.epil ], !dbg !6
  %cmp10.i = icmp sgt i32 %sum.0.i.lcssa, 17, !dbg !15
  %spec.select = select i1 %cmp10.i, i32 %sum.0.i.lcssa, i32 1, !dbg !15
  %cmp = icmp sgt i32 %n, 7, !dbg !16
  br i1 %cmp, label %if.then, label %if.end, !dbg !16

if.then:                                          ; preds = %for.end.i
  store i32 %spec.select, ptr addrspace(1) @gg, align 4, !dbg !17
  br label %if.end, !dbg !17

if.end:                                           ; preds = %if.then, %for.end.i
  ret void, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!1 = !DIFile(filename: "t3.cu", directory: "")
!2 = !{}
!3 = !{i32 1, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "kernel", linkageName: "_Z6kerneli", scope: !1, file: !1, line: 23, type: !5, scopeLine: 23, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !2)
!6 = !DILocation(line: 14, column: 3, scope: !7, inlinedAt: !9)
!7 = distinct !DILexicalBlock(scope: !8, file: !1, line: 12, column: 27)
!8 = distinct !DISubprogram(name: "C", linkageName: "_ZN1CC1Ev", scope: !1, file: !1, line: 12, type: !5, scopeLine: 12, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!9 = distinct !DILocation(line: 24, column: 3, scope: !10)
!10 = distinct !DILexicalBlock(scope: !4, file: !1, line: 23, column: 29)
!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.unroll.disable"}
!14 = distinct !{!14, !13}
!15 = !DILocation(line: 15, column: 3, scope: !7, inlinedAt: !9)
!16 = !DILocation(line: 25, column: 3, scope: !10)
!17 = !DILocation(line: 26, column: 5, scope: !18)
!18 = distinct !DILexicalBlock(scope: !10, file: !1, line: 25, column: 3)
!19 = !DILocation(line: 27, column: 1, scope: !10)
