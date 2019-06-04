; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Decorate {{[0-9]+}} UserSemantic "42"
; CHECK-SPIRV: Decorate {{[0-9]+}} UserSemantic "bar"
; CHECK-SPIRV: Decorate {{[0-9]+}} UserSemantic "{FOO}"
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 1 UserSemantic "128"
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 0 UserSemantic "{baz}"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

%class.anon = type { i8 }
%struct.bar = type { i32, i8 }

; CHECK-LLVM:  [[STR:@[a-zA-Z0-9_.]+]] = {{.*}}42
; CHECK-LLVM: [[STR2:@[a-zA-Z0-9_.]+]] = {{.*}}{FOO}
; CHECK-LLVM: [[STR3:@[a-zA-Z0-9_.]+]] = {{.*}}bar
; CHECK-LLVM: [[STR4:@[a-zA-Z0-9_.]+]] = {{.*}}{baz}
; CHECK-LLVM: [[STR5:@[a-zA-Z0-9_.]+]] = {{.*}}128
@.str = private unnamed_addr constant [3 x i8] c"42\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [23 x i8] c"annotate_attribute.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [6 x i8] c"{FOO}\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [4 x i8] c"bar\00", section "llvm.metadata"
@.str.4 = private unnamed_addr constant [6 x i8] c"{baz}\00", section "llvm.metadata"
@.str.5 = private unnamed_addr constant [4 x i8] c"128\00", section "llvm.metadata"

; Function Attrs: nounwind
define spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %class.anon, align 1
  %1 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #4
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %0)
  %2 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %2) #4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %this) #2 align 2 {
entry:
  %this.addr = alloca %class.anon*, align 8
  store %class.anon* %this, %class.anon** %this.addr, align 8, !tbaa !5
  %this1 = load %class.anon*, %class.anon** %this.addr, align 8
  call spir_func void @_Z3foov()
  call spir_func void @_Z3bazv()
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
define spir_func void @_Z3foov() #3 {
entry:
  %var_one = alloca i32, align 4
  %var_two = alloca i32, align 4
  %var_three = alloca i8, align 1
  %0 = bitcast i32* %var_one to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4
  %var_one1 = bitcast i32* %var_one to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* %[[VAR1:[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[STR]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_one1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 2)
  %1 = bitcast i32* %var_two to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #4
  %var_two2 = bitcast i32* %var_two to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* %[[VAR2:[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([6 x i8], [6 x i8]* [[STR2]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_two2, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 3)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %var_three) #4
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* %[[VAR3:[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[STR3]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_three, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 4)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %var_three) #4
  %2 = bitcast i32* %var_two to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #4
  %3 = bitcast i32* %var_one to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #4
  ret void
}

; Function Attrs: nounwind
declare void @llvm.var.annotation(i8*, i8*, i8*, i32) #4

; Function Attrs: nounwind
define spir_func void @_Z3bazv() #3 {
entry:
  %s1 = alloca %struct.bar, align 4
  %0 = bitcast %struct.bar* %s1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #4
  ; CHECK-LLVM: %[[FIELD1:.*]] = getelementptr inbounds %struct.bar, %struct.bar* %{{[a-zA-Z0-9]+}}, i32 0, i32 0
  ; CHECK-LLVM: %[[CAST1:.*]] = bitcast{{.*}}%[[FIELD1]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST1]]{{.*}}[[STR4]]
  %f1 = getelementptr inbounds %struct.bar, %struct.bar* %s1, i32 0, i32 0
  %1 = bitcast i32* %f1 to i8*
  %2 = call i8* @llvm.ptr.annotation.p0i8(i8* %1, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 8)
  %3 = bitcast i8* %2 to i32*
  store i32 0, i32* %3, align 4, !tbaa !9
  ; CHECK-LLVM: %[[FIELD2:.*]] = getelementptr inbounds %struct.bar, %struct.bar* %{{[a-zA-Z0-9]+}}, i32 0, i32 1
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[FIELD2]]{{.*}}[[STR5]]
  %f2 = getelementptr inbounds %struct.bar, %struct.bar* %s1, i32 0, i32 1
  %4 = call i8* @llvm.ptr.annotation.p0i8(i8* %f2, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.5, i32 0, i32 0), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.1, i32 0, i32 0), i32 9)
  store i8 0, i8* %4, align 4, !tbaa !12
  %5 = bitcast %struct.bar* %s1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %5) #4
  ret void
}

; Function Attrs: nounwind
declare i8* @llvm.ptr.annotation.p0i8(i8*, i8*, i8*, i32) #4

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind optnone noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 9.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !11, i64 0}
!10 = !{!"_ZTS3bar", !11, i64 0, !7, i64 4}
!11 = !{!"int", !7, i64 0}
!12 = !{!10, !7, i64 4}
