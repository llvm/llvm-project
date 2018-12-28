; RUN: opt -asfix %s -S -o - | FileCheck %s
; Compiled from:
;
; void foo(int * Data) {
;   int a = 10;
;   *Data = 1 + a;
;   *Data = 10;
; }
;
; void usages() {
;   __attribute__((address_space(1))) int *GLOB;
;   foo(GLOB);
; }
; ; ModuleID = 'new_test.cpp'
source_filename = "new_test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

%class.anon = type { i8 }

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
define spir_func void @_Z6usagesv() #3 {
entry:
  %GLOB = alloca i32 addrspace(1)*, align 8
  %0 = bitcast i32 addrspace(1)** %GLOB to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #4
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %GLOB, align 8, !tbaa !5
; CHECK: %[[CAST:.*]] = addrspacecast i32 addrspace(1)* %{{.*}} to i32 addrspace(4)*
  %2 = addrspacecast i32 addrspace(1)* %1 to i32*
; CHECK: call spir_func void @new.[[FOO:.*]](i32 addrspace(4)* %[[CAST]])
  call spir_func void @_Z3fooPi(i32* %2)
  %3 = bitcast i32 addrspace(1)** %GLOB to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %3) #4
  ret void
}

;CHECK: define spir_func void @new.[[FOO]](i32 addrspace(4)*)
; Function Attrs: nounwind
define spir_func void @_Z3fooPi(i32* %Data) #3 {
entry:
; CHECK: %[[DATA_ADDR:.*]] = alloca i32 addrspace(4)*
  %Data.addr = alloca i32*, align 8
; CHECK: %[[A:.*]] = alloca i32, align 4
  %a = alloca i32, align 4
; CHECK: store i32 addrspace(4)* %{{.*}}, i32 addrspace(4)** %[[DATA_ADDR]], align 8
  store i32* %Data, i32** %Data.addr, align 8, !tbaa !5
  %0 = bitcast i32* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4
  store i32 10, i32* %a, align 4, !tbaa !9
  %1 = load i32, i32* %a, align 4, !tbaa !9
; CHECK: %[[ADD:.*]] = add nsw i32 1, %{{.*}}
  %add = add nsw i32 1, %1
; CHECK: %[[DATA_LOAD:.*]] = load i32 addrspace(4)*, i32 addrspace(4)** %[[DATA_ADDR]]
  %2 = load i32*, i32** %Data.addr, align 8, !tbaa !5
; CHECK: store i32 %[[ADD]], i32 addrspace(4)* %[[DATA_LOAD]], align 4
  store i32 %add, i32* %2, align 4, !tbaa !9
; CHECK: %[[DATA_LOAD:.*]] = load i32 addrspace(4)*, i32 addrspace(4)** %[[DATA_ADDR]]
  %3 = load i32*, i32** %Data.addr, align 8, !tbaa !5
; CHECK: store i32 10, i32 addrspace(4)* %[[DATA_LOAD]], align 4
  store i32 10, i32* %3, align 4, !tbaa !9
  %4 = bitcast i32* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #4
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 8.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
