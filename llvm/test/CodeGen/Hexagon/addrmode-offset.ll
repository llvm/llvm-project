; RUN: llc -march=hexagon  -O3 < %s | FileCheck %s

; CHECK-NOT: [[REG0:(r[0-9]+)]] = memw([[REG0:(r[0-9]+)]]<<#2+##state-4)

%s.0 = type { i16, [10 x %s.1*] }
%s.1 = type { %s.2, i16, i16 }
%s.2 = type { i8, [15 x %s.3], [18 x %s.4], %s.5, i16 }
%s.3 = type { %s.5, %s.4*, i8*, i16, i8, i8, [3 x %s.4*], [3 x %s.4*], [3 x %s.4*] }
%s.4 = type { %s.5, %s.5*, i8, i16, i8 }
%s.5 = type { %s.5*, %s.5* }
%s.6 = type { i8, i8 }

@g0 = common global %s.0 zeroinitializer, align 4

; Function Attrs: nounwind optsize
define void @f0(%s.6* nocapture readonly %a0) local_unnamed_addr #0 {
b0:
  %v0 = bitcast %s.6* %a0 to %s.6*
  %v1 = getelementptr %s.6, %s.6* %v0, i32 0, i32 1
  %v2 = load i8, i8* %v1, align 1
  %v3 = zext i8 %v2 to i32
  %v4 = add nsw i32 %v3, -1
  %v5 = getelementptr %s.0, %s.0* @g0, i32 0, i32 1
  %v6 = getelementptr [10 x %s.1*], [10 x %s.1*]* %v5, i32 0, i32 %v4
  %v7 = load %s.1*, %s.1** %v6, align 4
  %v8 = icmp eq %s.1* %v7, null
  br i1 %v8, label %b4, label %b1

b1:                                               ; preds = %b0
  %v9 = bitcast %s.1* %v7 to %s.1*
  %v10 = bitcast %s.1* %v9 to i8*
  %v11 = load i8, i8* %v10, align 4
  %v12 = icmp eq i8 %v11, %v2
  br i1 %v12, label %b2, label %b4

b2:                                               ; preds = %b1
  %v13 = bitcast %s.6* %a0 to %s.6*
  tail call void @f1(%s.1* nonnull %v7) #2
  %v14 = getelementptr %s.6, %s.6* %v13, i32 0, i32 1
  %v15 = load i8, i8* %v14, align 1
  %v16 = zext i8 %v15 to i32
  %v17 = add nsw i32 %v16, -1
  %v18 = getelementptr [10 x %s.1*], [10 x %s.1*]* %v5, i32 0, i32 %v17
  %v19 = load %s.1*, %s.1** %v18, align 4
  %v20 = icmp eq %s.1* %v19, null
  br i1 %v20, label %b4, label %b3

b3:                                               ; preds = %b2
  %v21 = getelementptr %s.1, %s.1* %v19, i32 0, i32 0, i32 3
  tail call void @f2(%s.5* %v21) #2
  store %s.1* null, %s.1** %v18, align 4
  br label %b4

b4:                                               ; preds = %b3, %b2, %b1, %b0
  ret void
}

; Function Attrs: optsize
declare void @f1(%s.1*) #1

; Function Attrs: optsize
declare void @f2(%s.5*) #1

attributes #0 = { nounwind optsize "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { optsize "target-cpu"="hexagonv60" "target-features"="+hvx" }
attributes #2 = { nounwind }
