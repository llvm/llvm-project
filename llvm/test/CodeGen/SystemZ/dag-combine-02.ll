; Test that adjustSubwordCmp() maintains the chains properly when creating a
; new extending load.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -O3 | FileCheck %s

@g_56 = external hidden unnamed_addr global i64, align 8
@func_22.l_91 = external hidden unnamed_addr constant [4 x [7 x i16*]], align 8
@g_102 = external hidden unnamed_addr global i16**, align 8
@.str = external hidden unnamed_addr constant [2 x i8], align 2
@.str.1 = external hidden unnamed_addr constant [15 x i8], align 2
@crc32_context = external hidden unnamed_addr global i32, align 4
@crc32_tab = external hidden unnamed_addr global [256 x i32], align 4
@.str.2 = external hidden unnamed_addr constant [36 x i8], align 2
@.str.3 = external hidden unnamed_addr constant [15 x i8], align 2
@g_181.0.4.5 = external hidden unnamed_addr global i1, align 2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0

; Function Attrs: nounwind
define signext i32 @main(i32 signext, i8** nocapture readonly) local_unnamed_addr #1 {
  %3 = alloca [4 x [7 x i16*]], align 8
  %4 = icmp eq i32 %0, 2
  br i1 %4, label %5, label %11

; <label>:5:                                      ; preds = %2
  %6 = getelementptr inbounds i8*, i8** %1, i64 1
  %7 = load i8*, i8** %6, align 8
  %8 = tail call signext i32 @strcmp(i8* %7, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i64 0, i64 0)) #4
  %9 = icmp eq i32 %8, 0
  %10 = zext i1 %9 to i32
  br label %11

; <label>:11:                                     ; preds = %5, %2
  %12 = phi i32 [ 0, %2 ], [ %10, %5 ]
  br label %13

; <label>:13:                                     ; preds = %13, %11
  %14 = phi i64 [ 0, %11 ], [ %58, %13 ]
  %15 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %11 ], [ %59, %13 ]
  %16 = and <4 x i32> %15, <i32 1, i32 1, i32 1, i32 1>
  %17 = icmp eq <4 x i32> %16, zeroinitializer
  %18 = lshr <4 x i32> %15, <i32 1, i32 1, i32 1, i32 1>
  %19 = xor <4 x i32> %18, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %20 = select <4 x i1> %17, <4 x i32> %18, <4 x i32> %19
  %21 = and <4 x i32> %20, <i32 1, i32 1, i32 1, i32 1>
  %22 = icmp eq <4 x i32> %21, zeroinitializer
  %23 = lshr <4 x i32> %20, <i32 1, i32 1, i32 1, i32 1>
  %24 = xor <4 x i32> %23, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %25 = select <4 x i1> %22, <4 x i32> %23, <4 x i32> %24
  %26 = and <4 x i32> %25, <i32 1, i32 1, i32 1, i32 1>
  %27 = icmp eq <4 x i32> %26, zeroinitializer
  %28 = lshr <4 x i32> %25, <i32 1, i32 1, i32 1, i32 1>
  %29 = xor <4 x i32> %28, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %30 = select <4 x i1> %27, <4 x i32> %28, <4 x i32> %29
  %31 = and <4 x i32> %30, <i32 1, i32 1, i32 1, i32 1>
  %32 = icmp eq <4 x i32> %31, zeroinitializer
  %33 = lshr <4 x i32> %30, <i32 1, i32 1, i32 1, i32 1>
  %34 = xor <4 x i32> %33, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %35 = select <4 x i1> %32, <4 x i32> %33, <4 x i32> %34
  %36 = and <4 x i32> %35, <i32 1, i32 1, i32 1, i32 1>
  %37 = icmp eq <4 x i32> %36, zeroinitializer
  %38 = lshr <4 x i32> %35, <i32 1, i32 1, i32 1, i32 1>
  %39 = xor <4 x i32> %38, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %40 = select <4 x i1> %37, <4 x i32> %38, <4 x i32> %39
  %41 = and <4 x i32> %40, <i32 1, i32 1, i32 1, i32 1>
  %42 = icmp eq <4 x i32> %41, zeroinitializer
  %43 = lshr <4 x i32> %40, <i32 1, i32 1, i32 1, i32 1>
  %44 = xor <4 x i32> %43, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %45 = select <4 x i1> %42, <4 x i32> %43, <4 x i32> %44
  %46 = and <4 x i32> %45, <i32 1, i32 1, i32 1, i32 1>
  %47 = icmp eq <4 x i32> %46, zeroinitializer
  %48 = lshr <4 x i32> %45, <i32 1, i32 1, i32 1, i32 1>
  %49 = xor <4 x i32> %48, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %50 = select <4 x i1> %47, <4 x i32> %48, <4 x i32> %49
  %51 = and <4 x i32> %50, <i32 1, i32 1, i32 1, i32 1>
  %52 = icmp eq <4 x i32> %51, zeroinitializer
  %53 = lshr <4 x i32> %50, <i32 1, i32 1, i32 1, i32 1>
  %54 = xor <4 x i32> %53, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %55 = select <4 x i1> %52, <4 x i32> %53, <4 x i32> %54
  %56 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %14
  %57 = bitcast i32* %56 to <4 x i32>*
  store <4 x i32> %55, <4 x i32>* %57, align 4
  %58 = add i64 %14, 4
  %59 = add <4 x i32> %15, <i32 4, i32 4, i32 4, i32 4>
  %60 = icmp eq i64 %58, 256
  br i1 %60, label %61, label %13

; <label>:61:                                     ; preds = %13
; CHECK-LABEL: %bb.6:
; CHECK: stgrl   %r1, g_56
; CHECK: llhrl   %r1, g_56+6
; CHECK: stgrl   %r2, g_56
  store i64 0, i64* @g_56, align 8
  %62 = bitcast [4 x [7 x i16*]]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 224, i8* nonnull %62) #5
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %62, i8* bitcast ([4 x [7 x i16*]]* @func_22.l_91 to i8*), i64 224, i32 8, i1 false) #5
  %63 = getelementptr inbounds [4 x [7 x i16*]], [4 x [7 x i16*]]* %3, i64 0, i64 0, i64 2
  store i16** %63, i16*** @g_102, align 8
  %64 = load i64, i64* @g_56, align 8
  store i64 2, i64* @g_56, align 8
  %65 = and i64 %64, 65535
  %66 = icmp eq i64 %65, 0
  br i1 %66, label %68, label %67

; <label>:67:                                     ; preds = %61
  store i1 true, i1* @g_181.0.4.5, align 2
  br label %68

; <label>:68:                                     ; preds = %67, %61
  call void @llvm.lifetime.end.p0i8(i64 224, i8* nonnull %62) #5
  %69 = load i1, i1* @g_181.0.4.5, align 2
  %70 = select i1 %69, i32 0, i32 72
  %71 = load i32, i32* @crc32_context, align 4
  %72 = lshr i32 %71, 8
  %73 = and i32 %71, 255
  %74 = xor i32 %73, %70
  %75 = zext i32 %74 to i64
  %76 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %75
  %77 = load i32, i32* %76, align 4
  %78 = xor i32 %72, %77
  %79 = lshr i32 %78, 8
  %80 = and i32 %78, 255
  %81 = zext i32 %80 to i64
  %82 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %81
  %83 = load i32, i32* %82, align 4
  %84 = xor i32 %79, %83
  %85 = lshr i32 %84, 8
  %86 = and i32 %84, 255
  %87 = zext i32 %86 to i64
  %88 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %87
  %89 = load i32, i32* %88, align 4
  %90 = xor i32 %85, %89
  %91 = lshr i32 %90, 8
  %92 = and i32 %90, 255
  %93 = zext i32 %92 to i64
  %94 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %93
  %95 = load i32, i32* %94, align 4
  %96 = xor i32 %91, %95
  %97 = lshr i32 %96, 8
  %98 = and i32 %96, 255
  %99 = zext i32 %98 to i64
  %100 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %99
  %101 = load i32, i32* %100, align 4
  %102 = xor i32 %97, %101
  %103 = lshr i32 %102, 8
  %104 = and i32 %102, 255
  %105 = zext i32 %104 to i64
  %106 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %105
  %107 = load i32, i32* %106, align 4
  %108 = xor i32 %103, %107
  %109 = lshr i32 %108, 8
  %110 = and i32 %108, 255
  %111 = zext i32 %110 to i64
  %112 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %111
  %113 = load i32, i32* %112, align 4
  %114 = xor i32 %109, %113
  %115 = lshr i32 %114, 8
  %116 = and i32 %114, 255
  %117 = zext i32 %116 to i64
  %118 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %117
  %119 = load i32, i32* %118, align 4
  %120 = xor i32 %115, %119
  store i32 %120, i32* @crc32_context, align 4
  %121 = icmp eq i32 %12, 0
  br i1 %121, label %127, label %122

; <label>:122:                                    ; preds = %68
  %123 = xor i32 %120, -1
  %124 = zext i32 %123 to i64
  %125 = call signext i32 (i8*, ...) @printf(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str.2, i64 0, i64 0), i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.1, i64 0, i64 0), i64 %124) #5
  %126 = load i32, i32* @crc32_context, align 4
  br label %127

; <label>:127:                                    ; preds = %122, %68
  %128 = phi i32 [ %120, %68 ], [ %126, %122 ]
  %129 = xor i32 %128, -1
  %130 = call signext i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.3, i64 0, i64 0), i32 zeroext %129) #5
  ret i32 0
}

; Function Attrs: nounwind readonly
declare signext i32 @strcmp(i8* nocapture, i8* nocapture) local_unnamed_addr #2

; Function Attrs: nounwind
declare signext i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3
