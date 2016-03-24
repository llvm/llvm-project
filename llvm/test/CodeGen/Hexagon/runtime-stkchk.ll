; RUN: llc -march=hexagon -mcpu=hexagonv55 -enable-stackovf-sanitizer < %s | FileCheck %s

; CHECK-LABEL: foo_1
; CHECK: __runtime_stack_check
define i32 @foo_1(i32 %n) #0 {
entry:
  %local = alloca [1024 x i32], align 8
  %0 = bitcast [1024 x i32]* %local to i8*
  call void @llvm.lifetime.start(i64 4096, i8* %0) #1
  %arraydecay = getelementptr inbounds [1024 x i32], [1024 x i32]* %local, i32 0, i32 0
  call void @baz_1(i32* %arraydecay) #3
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32]* %local, i32 0, i32 %n
  %1 = load i32, i32* %arrayidx, align 4
  call void @llvm.lifetime.end(i64 4096, i8* %0) #1
  ret i32 %1
}

; CHECK-LABEL: foo_2
; CHECK: __save_r16_through_r19_stkchk
define i32 @foo_2(i32 %n, i32* %y) #0 {
entry:
  %local = alloca [2048 x i32], align 8
  %0 = bitcast [2048 x i32]* %local to i8*
  call void @llvm.lifetime.start(i64 8192, i8* %0) #1
  %arraydecay = getelementptr inbounds [2048 x i32], [2048 x i32]* %local, i32 0, i32 0
  call void @baz_2(i32* %y, i32* %arraydecay) #3
  %1 = load i32, i32* %y, align 4
  %add = add nsw i32 %n, %1
  %arrayidx = getelementptr inbounds [2048 x i32], [2048 x i32]* %local, i32 0, i32 %add
  %2 = load i32, i32* %arrayidx, align 4
  call void @llvm.lifetime.end(i64 8192, i8* %0) #1
  ret i32 %2
}

declare void @baz_1(i32*) #2
declare void @baz_2(i32*, i32*) #2
declare void @llvm.lifetime.start(i64, i8* nocapture) #1
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { optsize "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { optsize }

