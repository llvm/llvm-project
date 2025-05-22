; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mcpu=pwr8 -mtriple=powerpcle < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mcpu=pwr8 -mtriple=powerpc < %s | FileCheck %s
%struct.d = type { [131072 x i32] }

@a = dso_local local_unnamed_addr global [4096 x i32] zeroinitializer, align 4

; Function Attrs: mustprogress uwtable
define dso_local void @_Z1g1dILi17EE(ptr nocapture noundef readnone byval(%struct.d) align 4 %0) local_unnamed_addr #0 {
; CHECK-LABEL: _Z1g1dILi17EE:
; CHECK-NOT:    mtfprwz f0, r4
; CHECK-NOT:    stwx r3, r1, r4
; CHECK-NOT:    mffprwz r4, f0
; CHECK-NOT:    mtfprwz f0, r4
; CHECK-NOT:    lwzx r3, r1, r4
; CHECK-NOT:    mffprwz r4, f0
entry:
  %c = alloca %struct.d, align 4
  call void @llvm.lifetime.start.p0(i64 524288, ptr nonnull %c) #3
  br label %vector.body

vector.body:                                      ; preds = %vector.body.1, %entry
  %index = phi i32 [ 0, %entry ], [ %index.next.1, %vector.body.1 ]
  %vec.ind = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %entry ], [ %vec.ind.next.1, %vector.body.1 ]
  %step.add = add <4 x i32> %vec.ind, <i32 4, i32 4, i32 4, i32 4>
  %step.add21 = add <4 x i32> %vec.ind, <i32 8, i32 8, i32 8, i32 8>
  %step.add22 = add <4 x i32> %vec.ind, <i32 12, i32 12, i32 12, i32 12>
  %step.add23 = add <4 x i32> %vec.ind, <i32 16, i32 16, i32 16, i32 16>
  %step.add24 = add <4 x i32> %vec.ind, <i32 20, i32 20, i32 20, i32 20>
  %step.add25 = add <4 x i32> %vec.ind, <i32 24, i32 24, i32 24, i32 24>
  %step.add26 = add <4 x i32> %vec.ind, <i32 28, i32 28, i32 28, i32 28>
  %step.add27 = add <4 x i32> %vec.ind, <i32 32, i32 32, i32 32, i32 32>
  %step.add28 = add <4 x i32> %vec.ind, <i32 36, i32 36, i32 36, i32 36>
  %step.add29 = add <4 x i32> %vec.ind, <i32 40, i32 40, i32 40, i32 40>
  %step.add30 = add <4 x i32> %vec.ind, <i32 44, i32 44, i32 44, i32 44>
  %1 = getelementptr inbounds [4096 x i32], ptr @a, i32 0, i32 %index
  store <4 x i32> %vec.ind, ptr %1, align 4
  %2 = getelementptr inbounds i32, ptr %1, i32 4
  store <4 x i32> %step.add, ptr %2, align 4
  %3 = getelementptr inbounds i32, ptr %1, i32 8
  store <4 x i32> %step.add21, ptr %3, align 4
  %4 = getelementptr inbounds i32, ptr %1, i32 12
  store <4 x i32> %step.add22, ptr %4, align 4
  %5 = getelementptr inbounds i32, ptr %1, i32 16
  store <4 x i32> %step.add23, ptr %5, align 4
  %6 = getelementptr inbounds i32, ptr %1, i32 20
  store <4 x i32> %step.add24, ptr %6, align 4
  %7 = getelementptr inbounds i32, ptr %1, i32 24
  store <4 x i32> %step.add25, ptr %7, align 4
  %8 = getelementptr inbounds i32, ptr %1, i32 28
  store <4 x i32> %step.add26, ptr %8, align 4
  %9 = getelementptr inbounds i32, ptr %1, i32 32
  store <4 x i32> %step.add27, ptr %9, align 4
  %10 = getelementptr inbounds i32, ptr %1, i32 36
  store <4 x i32> %step.add28, ptr %10, align 4
  %11 = getelementptr inbounds i32, ptr %1, i32 40
  store <4 x i32> %step.add29, ptr %11, align 4
  %12 = getelementptr inbounds i32, ptr %1, i32 44
  store <4 x i32> %step.add30, ptr %12, align 4
  %index.next = add nuw nsw i32 %index, 48
  %13 = icmp eq i32 %index.next, 4080
  br i1 %13, label %for.body, label %vector.body.1

vector.body.1:                                    ; preds = %vector.body
  %vec.ind.next = add <4 x i32> %vec.ind, <i32 48, i32 48, i32 48, i32 48>
  %step.add.1 = add <4 x i32> %vec.ind, <i32 52, i32 52, i32 52, i32 52>
  %step.add21.1 = add <4 x i32> %vec.ind, <i32 56, i32 56, i32 56, i32 56>
  %step.add22.1 = add <4 x i32> %vec.ind, <i32 60, i32 60, i32 60, i32 60>
  %step.add23.1 = add <4 x i32> %vec.ind, <i32 64, i32 64, i32 64, i32 64>
  %step.add24.1 = add <4 x i32> %vec.ind, <i32 68, i32 68, i32 68, i32 68>
  %step.add25.1 = add <4 x i32> %vec.ind, <i32 72, i32 72, i32 72, i32 72>
  %step.add26.1 = add <4 x i32> %vec.ind, <i32 76, i32 76, i32 76, i32 76>
  %step.add27.1 = add <4 x i32> %vec.ind, <i32 80, i32 80, i32 80, i32 80>
  %step.add28.1 = add <4 x i32> %vec.ind, <i32 84, i32 84, i32 84, i32 84>
  %step.add29.1 = add <4 x i32> %vec.ind, <i32 88, i32 88, i32 88, i32 88>
  %step.add30.1 = add <4 x i32> %vec.ind, <i32 92, i32 92, i32 92, i32 92>
  %14 = getelementptr inbounds [4096 x i32], ptr @a, i32 0, i32 %index.next
  store <4 x i32> %vec.ind.next, ptr %14, align 4
  %15 = getelementptr inbounds i32, ptr %14, i32 4
  store <4 x i32> %step.add.1, ptr %15, align 4
  %16 = getelementptr inbounds i32, ptr %14, i32 8
  store <4 x i32> %step.add21.1, ptr %16, align 4
  %17 = getelementptr inbounds i32, ptr %14, i32 12
  store <4 x i32> %step.add22.1, ptr %17, align 4
  %18 = getelementptr inbounds i32, ptr %14, i32 16
  store <4 x i32> %step.add23.1, ptr %18, align 4
  %19 = getelementptr inbounds i32, ptr %14, i32 20
  store <4 x i32> %step.add24.1, ptr %19, align 4
  %20 = getelementptr inbounds i32, ptr %14, i32 24
  store <4 x i32> %step.add25.1, ptr %20, align 4
  %21 = getelementptr inbounds i32, ptr %14, i32 28
  store <4 x i32> %step.add26.1, ptr %21, align 4
  %22 = getelementptr inbounds i32, ptr %14, i32 32
  store <4 x i32> %step.add27.1, ptr %22, align 4
  %23 = getelementptr inbounds i32, ptr %14, i32 36
  store <4 x i32> %step.add28.1, ptr %23, align 4
  %24 = getelementptr inbounds i32, ptr %14, i32 40
  store <4 x i32> %step.add29.1, ptr %24, align 4
  %25 = getelementptr inbounds i32, ptr %14, i32 44
  store <4 x i32> %step.add30.1, ptr %25, align 4
  %index.next.1 = add nuw nsw i32 %index, 96
  %vec.ind.next.1 = add <4 x i32> %vec.ind, <i32 96, i32 96, i32 96, i32 96>
  br label %vector.body

vector.body37:                                    ; preds = %vector.body37.1, %for.body
  %index38 = phi i32 [ 0, %for.body ], [ %index.next53.1, %vector.body37.1 ]
  %vec.ind39 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %for.body ], [ %vec.ind.next52.1, %vector.body37.1 ]
  %step.add40 = add <4 x i32> %vec.ind39, <i32 4, i32 4, i32 4, i32 4>
  %step.add41 = add <4 x i32> %vec.ind39, <i32 8, i32 8, i32 8, i32 8>
  %step.add42 = add <4 x i32> %vec.ind39, <i32 12, i32 12, i32 12, i32 12>
  %step.add43 = add <4 x i32> %vec.ind39, <i32 16, i32 16, i32 16, i32 16>
  %step.add44 = add <4 x i32> %vec.ind39, <i32 20, i32 20, i32 20, i32 20>
  %step.add45 = add <4 x i32> %vec.ind39, <i32 24, i32 24, i32 24, i32 24>
  %step.add46 = add <4 x i32> %vec.ind39, <i32 28, i32 28, i32 28, i32 28>
  %step.add47 = add <4 x i32> %vec.ind39, <i32 32, i32 32, i32 32, i32 32>
  %step.add48 = add <4 x i32> %vec.ind39, <i32 36, i32 36, i32 36, i32 36>
  %step.add49 = add <4 x i32> %vec.ind39, <i32 40, i32 40, i32 40, i32 40>
  %step.add50 = add <4 x i32> %vec.ind39, <i32 44, i32 44, i32 44, i32 44>
  %26 = getelementptr inbounds [4096 x i32], ptr @a, i32 0, i32 %index38
  store <4 x i32> %vec.ind39, ptr %26, align 4
  %27 = getelementptr inbounds i32, ptr %26, i32 4
  store <4 x i32> %step.add40, ptr %27, align 4
  %28 = getelementptr inbounds i32, ptr %26, i32 8
  store <4 x i32> %step.add41, ptr %28, align 4
  %29 = getelementptr inbounds i32, ptr %26, i32 12
  store <4 x i32> %step.add42, ptr %29, align 4
  %30 = getelementptr inbounds i32, ptr %26, i32 16
  store <4 x i32> %step.add43, ptr %30, align 4
  %31 = getelementptr inbounds i32, ptr %26, i32 20
  store <4 x i32> %step.add44, ptr %31, align 4
  %32 = getelementptr inbounds i32, ptr %26, i32 24
  store <4 x i32> %step.add45, ptr %32, align 4
  %33 = getelementptr inbounds i32, ptr %26, i32 28
  store <4 x i32> %step.add46, ptr %33, align 4
  %34 = getelementptr inbounds i32, ptr %26, i32 32
  store <4 x i32> %step.add47, ptr %34, align 4
  %35 = getelementptr inbounds i32, ptr %26, i32 36
  store <4 x i32> %step.add48, ptr %35, align 4
  %36 = getelementptr inbounds i32, ptr %26, i32 40
  store <4 x i32> %step.add49, ptr %36, align 4
  %37 = getelementptr inbounds i32, ptr %26, i32 44
  store <4 x i32> %step.add50, ptr %37, align 4
  %index.next53 = add nuw nsw i32 %index38, 48
  %38 = icmp eq i32 %index.next53, 4080
  br i1 %38, label %for.body5, label %vector.body37.1

vector.body37.1:                                  ; preds = %vector.body37
  %vec.ind.next52 = add <4 x i32> %vec.ind39, <i32 48, i32 48, i32 48, i32 48>
  %step.add40.1 = add <4 x i32> %vec.ind39, <i32 52, i32 52, i32 52, i32 52>
  %step.add41.1 = add <4 x i32> %vec.ind39, <i32 56, i32 56, i32 56, i32 56>
  %step.add42.1 = add <4 x i32> %vec.ind39, <i32 60, i32 60, i32 60, i32 60>
  %step.add43.1 = add <4 x i32> %vec.ind39, <i32 64, i32 64, i32 64, i32 64>
  %step.add44.1 = add <4 x i32> %vec.ind39, <i32 68, i32 68, i32 68, i32 68>
  %step.add45.1 = add <4 x i32> %vec.ind39, <i32 72, i32 72, i32 72, i32 72>
  %step.add46.1 = add <4 x i32> %vec.ind39, <i32 76, i32 76, i32 76, i32 76>
  %step.add47.1 = add <4 x i32> %vec.ind39, <i32 80, i32 80, i32 80, i32 80>
  %step.add48.1 = add <4 x i32> %vec.ind39, <i32 84, i32 84, i32 84, i32 84>
  %step.add49.1 = add <4 x i32> %vec.ind39, <i32 88, i32 88, i32 88, i32 88>
  %step.add50.1 = add <4 x i32> %vec.ind39, <i32 92, i32 92, i32 92, i32 92>
  %39 = getelementptr inbounds [4096 x i32], ptr @a, i32 0, i32 %index.next53
  store <4 x i32> %vec.ind.next52, ptr %39, align 4
  %40 = getelementptr inbounds i32, ptr %39, i32 4
  store <4 x i32> %step.add40.1, ptr %40, align 4
  %41 = getelementptr inbounds i32, ptr %39, i32 8
  store <4 x i32> %step.add41.1, ptr %41, align 4
  %42 = getelementptr inbounds i32, ptr %39, i32 12
  store <4 x i32> %step.add42.1, ptr %42, align 4
  %43 = getelementptr inbounds i32, ptr %39, i32 16
  store <4 x i32> %step.add43.1, ptr %43, align 4
  %44 = getelementptr inbounds i32, ptr %39, i32 20
  store <4 x i32> %step.add44.1, ptr %44, align 4
  %45 = getelementptr inbounds i32, ptr %39, i32 24
  store <4 x i32> %step.add45.1, ptr %45, align 4
  %46 = getelementptr inbounds i32, ptr %39, i32 28
  store <4 x i32> %step.add46.1, ptr %46, align 4
  %47 = getelementptr inbounds i32, ptr %39, i32 32
  store <4 x i32> %step.add47.1, ptr %47, align 4
  %48 = getelementptr inbounds i32, ptr %39, i32 36
  store <4 x i32> %step.add48.1, ptr %48, align 4
  %49 = getelementptr inbounds i32, ptr %39, i32 40
  store <4 x i32> %step.add49.1, ptr %49, align 4
  %50 = getelementptr inbounds i32, ptr %39, i32 44
  store <4 x i32> %step.add50.1, ptr %50, align 4
  %index.next53.1 = add nuw nsw i32 %index38, 96
  %vec.ind.next52.1 = add <4 x i32> %vec.ind39, <i32 96, i32 96, i32 96, i32 96>
  br label %vector.body37

for.body:                                         ; preds = %vector.body
  store i32 4080, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4080), align 4
  store i32 4081, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4081), align 4
  store i32 4082, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4082), align 4
  store i32 4083, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4083), align 4
  store i32 4084, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4084), align 4
  store i32 4085, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4085), align 4
  store i32 4086, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4086), align 4
  store i32 4087, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4087), align 4
  store i32 4088, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4088), align 4
  store i32 4089, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4089), align 4
  store i32 4090, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4090), align 4
  store i32 4091, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4091), align 4
  store i32 4092, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4092), align 4
  store i32 4093, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4093), align 4
  store i32 4094, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4094), align 4
  store i32 4095, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4095), align 4
  call void @_ZN1dILi17EE1eEv(ptr noundef nonnull align 4 dereferenceable(524288) %c)
  br label %vector.body37

for.body5:                                        ; preds = %vector.body37
  store i32 4080, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4080), align 4
  store i32 4081, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4081), align 4
  store i32 4082, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4082), align 4
  store i32 4083, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4083), align 4
  store i32 4084, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4084), align 4
  store i32 4085, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4085), align 4
  store i32 4086, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4086), align 4
  store i32 4087, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4087), align 4
  store i32 4088, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4088), align 4
  store i32 4089, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4089), align 4
  store i32 4090, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4090), align 4
  store i32 4091, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4091), align 4
  store i32 4092, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4092), align 4
  store i32 4093, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4093), align 4
  store i32 4094, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4094), align 4
  store i32 4095, ptr getelementptr inbounds ([4096 x i32], ptr @a, i32 0, i32 4095), align 4
  call void @_Z1h1dILi17EE(ptr noundef nonnull byval(%struct.d) align 4 %c)
  call void @llvm.lifetime.end.p0(i64 524288, ptr nonnull %c) #3
  ret void
}

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

declare dso_local void @_ZN1dILi17EE1eEv(ptr noundef nonnull align 4 dereferenceable(524288)) local_unnamed_addr #2

declare dso_local void @_Z1h1dILi17EE(ptr noundef byval(%struct.d) align 4) local_unnamed_addr #2

attributes #0 = { nounwind }
