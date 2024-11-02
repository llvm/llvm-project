; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mcpu=pwr8 -mtriple=powerpcle < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mcpu=pwr8 -mtriple=powerpc < %s | FileCheck %s
%struct.d = type { [131072 x i32] }

@a = dso_local local_unnamed_addr global [4096 x i32] zeroinitializer, align 4

; Function Attrs: mustprogress uwtable
define dso_local void @_Z1g1dILi17EE(%struct.d* nocapture noundef readnone byval(%struct.d) align 4 %0) local_unnamed_addr #0 {
; CHECK-LABEL: _Z1g1dILi17EE:
; CHECK:    mtfprwz f0, r4
; CHECK:    stwx [[REG:r[0-9]+]], r1, r4
; CHECK:    mffprwz r4, f0
; CHECK:    mtfprwz f0, r4
; CHECK:    lwzx [[REG]], r1, r4
; CHECK:    mffprwz r4, f0
entry:
  %c = alloca %struct.d, align 4
  %1 = bitcast %struct.d* %c to i8*
  call void @llvm.lifetime.start.p0i8(i64 524288, i8* nonnull %1) #3
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
  %2 = getelementptr inbounds [4096 x i32], [4096 x i32]* @a, i32 0, i32 %index
  %3 = bitcast i32* %2 to <4 x i32>*
  store <4 x i32> %vec.ind, <4 x i32>* %3, align 4
  %4 = getelementptr inbounds i32, i32* %2, i32 4
  %5 = bitcast i32* %4 to <4 x i32>*
  store <4 x i32> %step.add, <4 x i32>* %5, align 4
  %6 = getelementptr inbounds i32, i32* %2, i32 8
  %7 = bitcast i32* %6 to <4 x i32>*
  store <4 x i32> %step.add21, <4 x i32>* %7, align 4
  %8 = getelementptr inbounds i32, i32* %2, i32 12
  %9 = bitcast i32* %8 to <4 x i32>*
  store <4 x i32> %step.add22, <4 x i32>* %9, align 4
  %10 = getelementptr inbounds i32, i32* %2, i32 16
  %11 = bitcast i32* %10 to <4 x i32>*
  store <4 x i32> %step.add23, <4 x i32>* %11, align 4
  %12 = getelementptr inbounds i32, i32* %2, i32 20
  %13 = bitcast i32* %12 to <4 x i32>*
  store <4 x i32> %step.add24, <4 x i32>* %13, align 4
  %14 = getelementptr inbounds i32, i32* %2, i32 24
  %15 = bitcast i32* %14 to <4 x i32>*
  store <4 x i32> %step.add25, <4 x i32>* %15, align 4
  %16 = getelementptr inbounds i32, i32* %2, i32 28
  %17 = bitcast i32* %16 to <4 x i32>*
  store <4 x i32> %step.add26, <4 x i32>* %17, align 4
  %18 = getelementptr inbounds i32, i32* %2, i32 32
  %19 = bitcast i32* %18 to <4 x i32>*
  store <4 x i32> %step.add27, <4 x i32>* %19, align 4
  %20 = getelementptr inbounds i32, i32* %2, i32 36
  %21 = bitcast i32* %20 to <4 x i32>*
  store <4 x i32> %step.add28, <4 x i32>* %21, align 4
  %22 = getelementptr inbounds i32, i32* %2, i32 40
  %23 = bitcast i32* %22 to <4 x i32>*
  store <4 x i32> %step.add29, <4 x i32>* %23, align 4
  %24 = getelementptr inbounds i32, i32* %2, i32 44
  %25 = bitcast i32* %24 to <4 x i32>*
  store <4 x i32> %step.add30, <4 x i32>* %25, align 4
  %index.next = add nuw nsw i32 %index, 48
  %26 = icmp eq i32 %index.next, 4080
  br i1 %26, label %for.body, label %vector.body.1

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
  %27 = getelementptr inbounds [4096 x i32], [4096 x i32]* @a, i32 0, i32 %index.next
  %28 = bitcast i32* %27 to <4 x i32>*
  store <4 x i32> %vec.ind.next, <4 x i32>* %28, align 4
  %29 = getelementptr inbounds i32, i32* %27, i32 4
  %30 = bitcast i32* %29 to <4 x i32>*
  store <4 x i32> %step.add.1, <4 x i32>* %30, align 4
  %31 = getelementptr inbounds i32, i32* %27, i32 8
  %32 = bitcast i32* %31 to <4 x i32>*
  store <4 x i32> %step.add21.1, <4 x i32>* %32, align 4
  %33 = getelementptr inbounds i32, i32* %27, i32 12
  %34 = bitcast i32* %33 to <4 x i32>*
  store <4 x i32> %step.add22.1, <4 x i32>* %34, align 4
  %35 = getelementptr inbounds i32, i32* %27, i32 16
  %36 = bitcast i32* %35 to <4 x i32>*
  store <4 x i32> %step.add23.1, <4 x i32>* %36, align 4
  %37 = getelementptr inbounds i32, i32* %27, i32 20
  %38 = bitcast i32* %37 to <4 x i32>*
  store <4 x i32> %step.add24.1, <4 x i32>* %38, align 4
  %39 = getelementptr inbounds i32, i32* %27, i32 24
  %40 = bitcast i32* %39 to <4 x i32>*
  store <4 x i32> %step.add25.1, <4 x i32>* %40, align 4
  %41 = getelementptr inbounds i32, i32* %27, i32 28
  %42 = bitcast i32* %41 to <4 x i32>*
  store <4 x i32> %step.add26.1, <4 x i32>* %42, align 4
  %43 = getelementptr inbounds i32, i32* %27, i32 32
  %44 = bitcast i32* %43 to <4 x i32>*
  store <4 x i32> %step.add27.1, <4 x i32>* %44, align 4
  %45 = getelementptr inbounds i32, i32* %27, i32 36
  %46 = bitcast i32* %45 to <4 x i32>*
  store <4 x i32> %step.add28.1, <4 x i32>* %46, align 4
  %47 = getelementptr inbounds i32, i32* %27, i32 40
  %48 = bitcast i32* %47 to <4 x i32>*
  store <4 x i32> %step.add29.1, <4 x i32>* %48, align 4
  %49 = getelementptr inbounds i32, i32* %27, i32 44
  %50 = bitcast i32* %49 to <4 x i32>*
  store <4 x i32> %step.add30.1, <4 x i32>* %50, align 4
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
  %51 = getelementptr inbounds [4096 x i32], [4096 x i32]* @a, i32 0, i32 %index38
  %52 = bitcast i32* %51 to <4 x i32>*
  store <4 x i32> %vec.ind39, <4 x i32>* %52, align 4
  %53 = getelementptr inbounds i32, i32* %51, i32 4
  %54 = bitcast i32* %53 to <4 x i32>*
  store <4 x i32> %step.add40, <4 x i32>* %54, align 4
  %55 = getelementptr inbounds i32, i32* %51, i32 8
  %56 = bitcast i32* %55 to <4 x i32>*
  store <4 x i32> %step.add41, <4 x i32>* %56, align 4
  %57 = getelementptr inbounds i32, i32* %51, i32 12
  %58 = bitcast i32* %57 to <4 x i32>*
  store <4 x i32> %step.add42, <4 x i32>* %58, align 4
  %59 = getelementptr inbounds i32, i32* %51, i32 16
  %60 = bitcast i32* %59 to <4 x i32>*
  store <4 x i32> %step.add43, <4 x i32>* %60, align 4
  %61 = getelementptr inbounds i32, i32* %51, i32 20
  %62 = bitcast i32* %61 to <4 x i32>*
  store <4 x i32> %step.add44, <4 x i32>* %62, align 4
  %63 = getelementptr inbounds i32, i32* %51, i32 24
  %64 = bitcast i32* %63 to <4 x i32>*
  store <4 x i32> %step.add45, <4 x i32>* %64, align 4
  %65 = getelementptr inbounds i32, i32* %51, i32 28
  %66 = bitcast i32* %65 to <4 x i32>*
  store <4 x i32> %step.add46, <4 x i32>* %66, align 4
  %67 = getelementptr inbounds i32, i32* %51, i32 32
  %68 = bitcast i32* %67 to <4 x i32>*
  store <4 x i32> %step.add47, <4 x i32>* %68, align 4
  %69 = getelementptr inbounds i32, i32* %51, i32 36
  %70 = bitcast i32* %69 to <4 x i32>*
  store <4 x i32> %step.add48, <4 x i32>* %70, align 4
  %71 = getelementptr inbounds i32, i32* %51, i32 40
  %72 = bitcast i32* %71 to <4 x i32>*
  store <4 x i32> %step.add49, <4 x i32>* %72, align 4
  %73 = getelementptr inbounds i32, i32* %51, i32 44
  %74 = bitcast i32* %73 to <4 x i32>*
  store <4 x i32> %step.add50, <4 x i32>* %74, align 4
  %index.next53 = add nuw nsw i32 %index38, 48
  %75 = icmp eq i32 %index.next53, 4080
  br i1 %75, label %for.body5, label %vector.body37.1

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
  %76 = getelementptr inbounds [4096 x i32], [4096 x i32]* @a, i32 0, i32 %index.next53
  %77 = bitcast i32* %76 to <4 x i32>*
  store <4 x i32> %vec.ind.next52, <4 x i32>* %77, align 4
  %78 = getelementptr inbounds i32, i32* %76, i32 4
  %79 = bitcast i32* %78 to <4 x i32>*
  store <4 x i32> %step.add40.1, <4 x i32>* %79, align 4
  %80 = getelementptr inbounds i32, i32* %76, i32 8
  %81 = bitcast i32* %80 to <4 x i32>*
  store <4 x i32> %step.add41.1, <4 x i32>* %81, align 4
  %82 = getelementptr inbounds i32, i32* %76, i32 12
  %83 = bitcast i32* %82 to <4 x i32>*
  store <4 x i32> %step.add42.1, <4 x i32>* %83, align 4
  %84 = getelementptr inbounds i32, i32* %76, i32 16
  %85 = bitcast i32* %84 to <4 x i32>*
  store <4 x i32> %step.add43.1, <4 x i32>* %85, align 4
  %86 = getelementptr inbounds i32, i32* %76, i32 20
  %87 = bitcast i32* %86 to <4 x i32>*
  store <4 x i32> %step.add44.1, <4 x i32>* %87, align 4
  %88 = getelementptr inbounds i32, i32* %76, i32 24
  %89 = bitcast i32* %88 to <4 x i32>*
  store <4 x i32> %step.add45.1, <4 x i32>* %89, align 4
  %90 = getelementptr inbounds i32, i32* %76, i32 28
  %91 = bitcast i32* %90 to <4 x i32>*
  store <4 x i32> %step.add46.1, <4 x i32>* %91, align 4
  %92 = getelementptr inbounds i32, i32* %76, i32 32
  %93 = bitcast i32* %92 to <4 x i32>*
  store <4 x i32> %step.add47.1, <4 x i32>* %93, align 4
  %94 = getelementptr inbounds i32, i32* %76, i32 36
  %95 = bitcast i32* %94 to <4 x i32>*
  store <4 x i32> %step.add48.1, <4 x i32>* %95, align 4
  %96 = getelementptr inbounds i32, i32* %76, i32 40
  %97 = bitcast i32* %96 to <4 x i32>*
  store <4 x i32> %step.add49.1, <4 x i32>* %97, align 4
  %98 = getelementptr inbounds i32, i32* %76, i32 44
  %99 = bitcast i32* %98 to <4 x i32>*
  store <4 x i32> %step.add50.1, <4 x i32>* %99, align 4
  %index.next53.1 = add nuw nsw i32 %index38, 96
  %vec.ind.next52.1 = add <4 x i32> %vec.ind39, <i32 96, i32 96, i32 96, i32 96>
  br label %vector.body37

for.body:                                         ; preds = %vector.body
  store i32 4080, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4080), align 4
  store i32 4081, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4081), align 4
  store i32 4082, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4082), align 4
  store i32 4083, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4083), align 4
  store i32 4084, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4084), align 4
  store i32 4085, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4085), align 4
  store i32 4086, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4086), align 4
  store i32 4087, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4087), align 4
  store i32 4088, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4088), align 4
  store i32 4089, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4089), align 4
  store i32 4090, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4090), align 4
  store i32 4091, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4091), align 4
  store i32 4092, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4092), align 4
  store i32 4093, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4093), align 4
  store i32 4094, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4094), align 4
  store i32 4095, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4095), align 4
  call void @_ZN1dILi17EE1eEv(%struct.d* noundef nonnull align 4 dereferenceable(524288) %c)
  br label %vector.body37

for.body5:                                        ; preds = %vector.body37
  store i32 4080, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4080), align 4
  store i32 4081, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4081), align 4
  store i32 4082, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4082), align 4
  store i32 4083, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4083), align 4
  store i32 4084, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4084), align 4
  store i32 4085, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4085), align 4
  store i32 4086, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4086), align 4
  store i32 4087, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4087), align 4
  store i32 4088, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4088), align 4
  store i32 4089, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4089), align 4
  store i32 4090, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4090), align 4
  store i32 4091, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4091), align 4
  store i32 4092, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4092), align 4
  store i32 4093, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4093), align 4
  store i32 4094, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4094), align 4
  store i32 4095, i32* getelementptr inbounds ([4096 x i32], [4096 x i32]* @a, i32 0, i32 4095), align 4
  call void @_Z1h1dILi17EE(%struct.d* noundef nonnull byval(%struct.d) align 4 %c)
  call void @llvm.lifetime.end.p0i8(i64 524288, i8* nonnull %1) #3
  ret void
}

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local void @_ZN1dILi17EE1eEv(%struct.d* noundef nonnull align 4 dereferenceable(524288)) local_unnamed_addr #2

declare dso_local void @_Z1h1dILi17EE(%struct.d* noundef byval(%struct.d) align 4) local_unnamed_addr #2

attributes #0 = { nounwind }
