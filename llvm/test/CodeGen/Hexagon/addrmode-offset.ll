; RUN: llc -march=hexagon  -O3 < %s | FileCheck %s

; CHECK-NOT: [[REG0:(r[0-9]+)]] = memw([[REG0:(r[0-9]+)]]<<#2+##state-4)

%s.0 = type { i16, [10 x ptr] }
%s.1 = type { %s.2, i16, i16 }
%s.2 = type { i8, [15 x %s.3], [18 x %s.4], %s.5, i16 }
%s.3 = type { %s.5, ptr, ptr, i16, i8, i8, [3 x ptr], [3 x ptr], [3 x ptr] }
%s.4 = type { %s.5, ptr, i8, i16, i8 }
%s.5 = type { ptr, ptr }
%s.6 = type { i8, i8 }

@g0 = common global %s.0 zeroinitializer, align 4

; Function Attrs: nounwind optsize
define void @f0(ptr nocapture readonly %a0) local_unnamed_addr #0 {
b0:
  %v1 = getelementptr %s.6, ptr %a0, i32 0, i32 1
  %v2 = load i8, ptr %v1, align 1
  %v3 = zext i8 %v2 to i32
  %v4 = add nsw i32 %v3, -1
  %v5 = getelementptr %s.0, ptr @g0, i32 0, i32 1
  %v6 = getelementptr [10 x ptr], ptr %v5, i32 0, i32 %v4
  %v7 = load ptr, ptr %v6, align 4
  %v8 = icmp eq ptr %v7, null
  br i1 %v8, label %b4, label %b1

b1:                                               ; preds = %b0
  %v11 = load i8, ptr %v7, align 4
  %v12 = icmp eq i8 %v11, %v2
  br i1 %v12, label %b2, label %b4

b2:                                               ; preds = %b1
  tail call void @f1(ptr nonnull %v7) #2
  %v14 = getelementptr %s.6, ptr %a0, i32 0, i32 1
  %v15 = load i8, ptr %v14, align 1
  %v16 = zext i8 %v15 to i32
  %v17 = add nsw i32 %v16, -1
  %v18 = getelementptr [10 x ptr], ptr %v5, i32 0, i32 %v17
  %v19 = load ptr, ptr %v18, align 4
  %v20 = icmp eq ptr %v19, null
  br i1 %v20, label %b4, label %b3

b3:                                               ; preds = %b2
  %v21 = getelementptr %s.1, ptr %v19, i32 0, i32 0, i32 3
  tail call void @f2(ptr %v21) #2
  store ptr null, ptr %v18, align 4
  br label %b4

b4:                                               ; preds = %b3, %b2, %b1, %b0
  ret void
}

; Function Attrs: optsize
declare void @f1(ptr) #1

; Function Attrs: optsize
declare void @f2(ptr) #1

attributes #0 = { nounwind optsize "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { optsize "target-cpu"="hexagonv60" "target-features"="+hvx" }
attributes #2 = { nounwind }
