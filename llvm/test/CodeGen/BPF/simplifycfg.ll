; RUN: opt -O2 -S < %s | FileCheck %s
;
; This test tries to ensure that simplifycfg hoisting common instructions
; of then/else branch indeed happens. BPF target has added an IR pass
; before loop optimizations as Commit 1d51dc38d89b
; ([SimplifyCFG][LoopRotate] SimplifyCFG: disable common instruction
;  hoisting by default, enable late in pipeline)
; disabled common instruction hoisting. Due to optimization triggered
; code changes, later SimplifyCFG may not be able to perform optimization
; even common inst hoisting is enabled.
;
; Source:
;   typedef struct {
;     void *f_back;
;   } FrameData;
;   extern int get_data(void *, void *);
;   extern void get_frame_ptr(void *);
;   int test() {
;     void *frame_ptr;
;     FrameData frame;
;
;     get_frame_ptr(&frame_ptr);
;
;     #pragma nounroll
;     for (int i = 0; i < 6; i++) {
;       if (frame_ptr && get_data(frame_ptr, &frame)) {
;         frame_ptr = frame.f_back;
;       }
;     }
;     return frame_ptr == 0;
;   }
; Compilation flag:
;   clang -target bpf -O2 -Xclang -disable-llvm-passes -S -emit-llvm t.c -o t.ll

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

%struct.FrameData = type { i8* }

; Function Attrs: nounwind
define dso_local i32 @test() #0 {
entry:
  %frame_ptr = alloca i8*, align 8
  %frame = alloca %struct.FrameData, align 8
  %i = alloca i32, align 4
  %0 = bitcast i8** %frame_ptr to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #3
  %1 = bitcast %struct.FrameData* %frame to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %1) #3
  %2 = bitcast i8** %frame_ptr to i8*
  call void @get_frame_ptr(i8* %2)
  %3 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #3
  store i32 0, i32* %i, align 4, !tbaa !2
  br label %for.cond

; CHECK-LABEL:    entry
; CHECK:          %{{[0-9]+}} = load i8*, i8** %frame_ptr, align 8
; CHECK:          %{{[0-9a-z.]+}} = icmp eq i8* %2, null
; CHECK:          br label

for.cond:                                         ; preds = %for.inc, %entry
  %4 = load i32, i32* %i, align 4, !tbaa !2
  %cmp = icmp slt i32 %4, 6
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %5 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %5) #3
  br label %for.end

for.body:                                         ; preds = %for.cond
  %6 = load i8*, i8** %frame_ptr, align 8, !tbaa !6
  %tobool = icmp ne i8* %6, null
  br i1 %tobool, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %for.body
  %7 = load i8*, i8** %frame_ptr, align 8, !tbaa !6
  %8 = bitcast %struct.FrameData* %frame to i8*
  %call = call i32 @get_data(i8* %7, i8* %8)
  %tobool1 = icmp ne i32 %call, 0
  br i1 %tobool1, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %f_back = getelementptr inbounds %struct.FrameData, %struct.FrameData* %frame, i32 0, i32 0
  %9 = load i8*, i8** %f_back, align 8, !tbaa !8
  store i8* %9, i8** %frame_ptr, align 8, !tbaa !6
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %10 = load i32, i32* %i, align 4, !tbaa !2
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %i, align 4, !tbaa !2
  br label %for.cond, !llvm.loop !10

for.end:                                          ; preds = %for.cond.cleanup
  %11 = load i8*, i8** %frame_ptr, align 8, !tbaa !6
  %cmp2 = icmp eq i8* %11, null
  %conv = zext i1 %cmp2 to i32
  %12 = bitcast %struct.FrameData* %frame to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %12) #3
  %13 = bitcast i8** %frame_ptr to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %13) #3
  ret i32 %conv
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local void @get_frame_ptr(i8*) #2

declare dso_local i32 @get_data(i8*, i8*) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git 1b3c1c543269da36ae41ab84f646cf98d2e5b1e5)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!8 = !{!9, !7, i64 0}
!9 = !{!"", !7, i64 0}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.unroll.disable"}
