; RUN: opt -O2 -S -mtriple=bpf-pc-linux %s -o %t1
; RUN: llc %t1 -mcpu=v2 -o - | FileCheck %s
; RUN: opt -O2 -S -mtriple=bpf-pc-linux %s -o %t1
; RUN: llc %t1 -mcpu=v3 -o - | FileCheck %s
;
; Source:
;   struct buf {
;     unsigned int len;
;     unsigned char data[16];
;   };
;
;   long copy_chunk(void *ctx, unsigned int off, void *dst, unsigned int size);
;
;   int foo(void *ctx, unsigned short raw_len) {
;     struct buf b = {};
;     unsigned short n = raw_len - 1;
;     if (n >= 16) {
;       b.len = 16;
;       if (copy_chunk(ctx, 0, b.data, 16) != 0)
;         return -1;
;     } else {
;       b.len = n;
;       for (int i = 0; i < 16; ++i) {
;         if (i >= n)
;           break;
;         if (copy_chunk(ctx, i, b.data + i, 1) != 0)
;           return -1;
;       }
;     }
;     return b.len + b.data[0];
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -Xclang -disable-llvm-passes repro.c

%struct.buf = type { i32, [16 x i8] }

; Function Attrs: nounwind
define dso_local i32 @foo(ptr noundef %0, i16 noundef zeroext %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i16, align 2
  %6 = alloca %struct.buf, align 4
  %7 = alloca i16, align 2
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store ptr %0, ptr %4, align 8, !tbaa !3
  store i16 %1, ptr %5, align 2, !tbaa !7
  call void @llvm.lifetime.start.p0(i64 20, ptr %6) #4
  call void @llvm.memset.p0.i64(ptr align 4 %6, i8 0, i64 20, i1 false)
  call void @llvm.lifetime.start.p0(i64 2, ptr %7) #4
  %10 = load i16, ptr %5, align 2, !tbaa !7
  %11 = zext i16 %10 to i32
  %12 = sub nsw i32 %11, 1
  %13 = trunc i32 %12 to i16
  store i16 %13, ptr %7, align 2, !tbaa !7
  %14 = load i16, ptr %7, align 2, !tbaa !7
  %15 = zext i16 %14 to i32
  %16 = icmp sge i32 %15, 16
  br i1 %16, label %17, label %26

17:                                               ; preds = %2
  %18 = getelementptr inbounds nuw %struct.buf, ptr %6, i32 0, i32 0
  store i32 16, ptr %18, align 4, !tbaa !9
  %19 = load ptr, ptr %4, align 8, !tbaa !3
  %20 = getelementptr inbounds nuw %struct.buf, ptr %6, i32 0, i32 1
  %21 = getelementptr inbounds [16 x i8], ptr %20, i64 0, i64 0
  %22 = call i64 @copy_chunk(ptr noundef %19, i32 noundef 0, ptr noundef %21, i32 noundef 16)
  %23 = icmp ne i64 %22, 0
  br i1 %23, label %24, label %25

24:                                               ; preds = %17
  store i32 -1, ptr %3, align 4
  store i32 1, ptr %8, align 4
  br label %66

25:                                               ; preds = %17
  br label %58

26:                                               ; preds = %2
  %27 = load i16, ptr %7, align 2, !tbaa !7
  %28 = zext i16 %27 to i32
  %29 = getelementptr inbounds nuw %struct.buf, ptr %6, i32 0, i32 0
  store i32 %28, ptr %29, align 4, !tbaa !9
  call void @llvm.lifetime.start.p0(i64 4, ptr %9) #4
  store i32 0, ptr %9, align 4, !tbaa !12
  br label %30

30:                                               ; preds = %52, %26
  %31 = load i32, ptr %9, align 4, !tbaa !12
  %32 = icmp slt i32 %31, 16
  br i1 %32, label %34, label %33

33:                                               ; preds = %30
  store i32 2, ptr %8, align 4
  br label %55

34:                                               ; preds = %30
  %35 = load i32, ptr %9, align 4, !tbaa !12
  %36 = load i16, ptr %7, align 2, !tbaa !7
  %37 = zext i16 %36 to i32
  %38 = icmp sge i32 %35, %37
  br i1 %38, label %39, label %40

39:                                               ; preds = %34
  store i32 2, ptr %8, align 4
  br label %55

40:                                               ; preds = %34
  %41 = load ptr, ptr %4, align 8, !tbaa !3
  %42 = load i32, ptr %9, align 4, !tbaa !12
  %43 = getelementptr inbounds nuw %struct.buf, ptr %6, i32 0, i32 1
  %44 = getelementptr inbounds [16 x i8], ptr %43, i64 0, i64 0
  %45 = load i32, ptr %9, align 4, !tbaa !12
  %46 = sext i32 %45 to i64
  %47 = getelementptr inbounds i8, ptr %44, i64 %46
  %48 = call i64 @copy_chunk(ptr noundef %41, i32 noundef %42, ptr noundef %47, i32 noundef 1)
  %49 = icmp ne i64 %48, 0
  br i1 %49, label %50, label %51

50:                                               ; preds = %40
  store i32 -1, ptr %3, align 4
  store i32 1, ptr %8, align 4
  br label %55

51:                                               ; preds = %40
  br label %52

52:                                               ; preds = %51
  %53 = load i32, ptr %9, align 4, !tbaa !12
  %54 = add nsw i32 %53, 1
  store i32 %54, ptr %9, align 4, !tbaa !12
  br label %30, !llvm.loop !13

55:                                               ; preds = %50, %39, %33
  call void @llvm.lifetime.end.p0(i64 4, ptr %9) #4
  %56 = load i32, ptr %8, align 4
  switch i32 %56, label %66 [
    i32 2, label %57
  ]

57:                                               ; preds = %55
  br label %58

58:                                               ; preds = %57, %25
  %59 = getelementptr inbounds nuw %struct.buf, ptr %6, i32 0, i32 0
  %60 = load i32, ptr %59, align 4, !tbaa !9
  %61 = getelementptr inbounds nuw %struct.buf, ptr %6, i32 0, i32 1
  %62 = getelementptr inbounds [16 x i8], ptr %61, i64 0, i64 0
  %63 = load i8, ptr %62, align 4, !tbaa !15
  %64 = zext i8 %63 to i32
  %65 = add i32 %60, %64
  store i32 %65, ptr %3, align 4
  store i32 1, ptr %8, align 4
  br label %66

66:                                               ; preds = %58, %55, %24
  call void @llvm.lifetime.end.p0(i64 2, ptr %7) #4
  call void @llvm.lifetime.end.p0(i64 20, ptr %6) #4
  %67 = load i32, ptr %3, align 4
  ret i32 %67
}

; CHECK-LABEL: foo:
; CHECK:       .LBB0_5:
; CHECK-NEXT:    r[[REG:[0-9]+]] += 1
; CHECK-NEXT:    if r[[REG]] >= r{{[0-9]+}} goto .LBB0_7
; CHECK-NEXT:  .LBB0_6:
; CHECK:         call copy_chunk

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

declare dso_local i64 @copy_chunk(ptr noundef, i32 noundef, ptr noundef, i32 noundef) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"Debian clang version 21.1.8 (3)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !5, i64 0}
!9 = !{!10, !11, i64 0}
!10 = !{!"buf", !11, i64 0, !5, i64 4}
!11 = !{!"int", !5, i64 0}
!12 = !{!11, !11, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!5, !5, i64 0}
