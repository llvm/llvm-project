; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -O2 < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -O2 -stats -o /dev/null < %s 2>&1 \
; RUN:   | FileCheck --check-prefix=STATS %s

; SHA-512 message-schedule recurrence from #209946.
; CHECK-LABEL: schedule:
; CHECK:       loop0(.LBB{{[0-9]+}}_1,#15)
; CHECK:       .LBB{{[0-9]+}}_1:
; CHECK:       memd(r{{[0-9]+}}+#48) = r{{[0-9]+}}:{{[0-9]+}}
; CHECK:       memd(r{{[0-9]+}}+#56) = r{{[0-9]+}}:{{[0-9]+}}
; CHECK:       memd(r{{[0-9]+}}+#64) = r{{[0-9]+}}:{{[0-9]+}}
; CHECK:       memd(r{{[0-9]+}}+#72) = r{{[0-9]+}}:{{[0-9]+}}
; CHECK:       } :endloop0

; STATS-NOT: 1 pipeliner   - Number of loops software pipelined

; ModuleID = 'swp-sha512.c'
source_filename = "swp-sha512.c"
target triple = "hexagon-unknown-linux-musl"

define void @schedule(ptr %W) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.066 = phi i32 [ 16, %entry ], [ %inc.3, %for.body ]
  %0 = getelementptr i64, ptr %W, i32 %i.066
  %arrayidx = getelementptr i8, ptr %0, i32 -120
  %1 = load i64, ptr %arrayidx, align 8
  %or = call i64 @llvm.fshl.i64(i64 %1, i64 %1, i64 63)
  %or9 = call i64 @llvm.fshl.i64(i64 %1, i64 %1, i64 56)
  %xor = xor i64 %or, %or9
  %shr12 = lshr i64 %1, 7
  %xor13 = xor i64 %xor, %shr12
  %arrayidx15 = getelementptr i8, ptr %0, i32 -16
  %2 = load i64, ptr %arrayidx15, align 8
  %or20 = call i64 @llvm.fshl.i64(i64 %2, i64 %2, i64 45)
  %or27 = call i64 @llvm.fshl.i64(i64 %2, i64 %2, i64 3)
  %xor28 = xor i64 %or20, %or27
  %shr31 = lshr i64 %2, 6
  %xor32 = xor i64 %xor28, %shr31
  %arrayidx34 = getelementptr i8, ptr %0, i32 -128
  %3 = load i64, ptr %arrayidx34, align 8
  %add = add i64 %xor13, %3
  %arrayidx36 = getelementptr i8, ptr %0, i32 -56
  %4 = load i64, ptr %arrayidx36, align 8
  %add37 = add i64 %add, %4
  %add38 = add i64 %add37, %xor32
  store i64 %add38, ptr %0, align 8
  %5 = getelementptr i64, ptr %W, i32 %i.066
  %6 = getelementptr i8, ptr %5, i32 8
  %arrayidx.1 = getelementptr i8, ptr %5, i32 -112
  %7 = load i64, ptr %arrayidx.1, align 8
  %or.1 = call i64 @llvm.fshl.i64(i64 %7, i64 %7, i64 63)
  %or9.1 = call i64 @llvm.fshl.i64(i64 %7, i64 %7, i64 56)
  %xor.1 = xor i64 %or.1, %or9.1
  %shr12.1 = lshr i64 %7, 7
  %xor13.1 = xor i64 %xor.1, %shr12.1
  %arrayidx15.1 = getelementptr i8, ptr %5, i32 -8
  %8 = load i64, ptr %arrayidx15.1, align 8
  %or20.1 = call i64 @llvm.fshl.i64(i64 %8, i64 %8, i64 45)
  %or27.1 = call i64 @llvm.fshl.i64(i64 %8, i64 %8, i64 3)
  %xor28.1 = xor i64 %or20.1, %or27.1
  %shr31.1 = lshr i64 %8, 6
  %xor32.1 = xor i64 %xor28.1, %shr31.1
  %add.1 = add i64 %xor13.1, %1
  %arrayidx36.1 = getelementptr i8, ptr %5, i32 -48
  %9 = load i64, ptr %arrayidx36.1, align 8
  %add37.1 = add i64 %add.1, %9
  %add38.1 = add i64 %add37.1, %xor32.1
  store i64 %add38.1, ptr %6, align 8
  %10 = getelementptr i64, ptr %W, i32 %i.066
  %11 = getelementptr i8, ptr %10, i32 16
  %arrayidx.2 = getelementptr i8, ptr %10, i32 -104
  %12 = load i64, ptr %arrayidx.2, align 8
  %or.2 = call i64 @llvm.fshl.i64(i64 %12, i64 %12, i64 63)
  %or9.2 = call i64 @llvm.fshl.i64(i64 %12, i64 %12, i64 56)
  %xor.2 = xor i64 %or.2, %or9.2
  %shr12.2 = lshr i64 %12, 7
  %xor13.2 = xor i64 %xor.2, %shr12.2
  %13 = load i64, ptr %10, align 8
  %or20.2 = call i64 @llvm.fshl.i64(i64 %13, i64 %13, i64 45)
  %or27.2 = call i64 @llvm.fshl.i64(i64 %13, i64 %13, i64 3)
  %xor28.2 = xor i64 %or20.2, %or27.2
  %shr31.2 = lshr i64 %13, 6
  %xor32.2 = xor i64 %xor28.2, %shr31.2
  %add.2 = add i64 %xor13.2, %7
  %arrayidx36.2 = getelementptr i8, ptr %10, i32 -40
  %14 = load i64, ptr %arrayidx36.2, align 8
  %add37.2 = add i64 %add.2, %14
  %add38.2 = add i64 %add37.2, %xor32.2
  store i64 %add38.2, ptr %11, align 8
  %15 = getelementptr i64, ptr %W, i32 %i.066
  %16 = getelementptr i8, ptr %15, i32 24
  %arrayidx.3 = getelementptr i8, ptr %15, i32 -96
  %17 = load i64, ptr %arrayidx.3, align 8
  %or.3 = call i64 @llvm.fshl.i64(i64 %17, i64 %17, i64 63)
  %or9.3 = call i64 @llvm.fshl.i64(i64 %17, i64 %17, i64 56)
  %xor.3 = xor i64 %or.3, %or9.3
  %shr12.3 = lshr i64 %17, 7
  %xor13.3 = xor i64 %xor.3, %shr12.3
  %arrayidx15.3 = getelementptr i8, ptr %15, i32 8
  %18 = load i64, ptr %arrayidx15.3, align 8
  %or20.3 = call i64 @llvm.fshl.i64(i64 %18, i64 %18, i64 45)
  %or27.3 = call i64 @llvm.fshl.i64(i64 %18, i64 %18, i64 3)
  %xor28.3 = xor i64 %or20.3, %or27.3
  %shr31.3 = lshr i64 %18, 6
  %xor32.3 = xor i64 %xor28.3, %shr31.3
  %add.3 = add i64 %xor13.3, %12
  %arrayidx36.3 = getelementptr i8, ptr %15, i32 -32
  %19 = load i64, ptr %arrayidx36.3, align 8
  %add37.3 = add i64 %add.3, %19
  %add38.3 = add i64 %add37.3, %xor32.3
  store i64 %add38.3, ptr %16, align 8
  %inc.3 = add i32 %i.066, 4
  %exitcond.not.3 = icmp eq i32 %inc.3, 80
  br i1 %exitcond.not.3, label %for.cond.cleanup, label %for.body
}

declare i64 @llvm.fshl.i64(i64, i64, i64)
