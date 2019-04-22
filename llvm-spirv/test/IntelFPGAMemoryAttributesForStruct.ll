; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 1 RegisterINTEL 1
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 0 MemoryINTEL "DEFAULT"
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 3 MemoryINTEL "DEFAULT"
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 2 MemoryINTEL "MLAB"
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 0 NumbanksINTEL 4
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 3 BankwidthINTEL 8
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 4 MaxconcurrencyINTEL 4

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

%class.anon = type { i8 }
%struct.foo = type { i32, i32, i32, i32, i8 }

; CHECK-LLVM: [[STR1:@[a-zA-Z0-9_.]+]] = {{.*}}{memory:DEFAULT}{numbanks:4}
; CHECK-LLVM: [[STR2:@[a-zA-Z0-9_.]+]] = {{.*}}{register:1}
; CHECK-LLVM: [[STR3:@[a-zA-Z0-9_.]+]] = {{.*}}{memory:MLAB}
; CHECK-LLVM: [[STR4:@[a-zA-Z0-9_.]+]] = {{.*}}{memory:DEFAULT}{bankwidth:8}
; CHECK-LLVM: [[STR5:@[a-zA-Z0-9_.]+]] = {{.*}}{memory:DEFAULT}{max_concurrency:4}
@.str = private unnamed_addr constant [29 x i8] c"{memory:DEFAULT}{numbanks:4}\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [16 x i8] c"test_struct.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [13 x i8] c"{register:1}\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [14 x i8] c"{memory:MLAB}\00", section "llvm.metadata"
@.str.4 = private unnamed_addr constant [30 x i8] c"{memory:DEFAULT}{bankwidth:8}\00", section "llvm.metadata"
@.str.5 = private unnamed_addr constant [36 x i8] c"{memory:DEFAULT}{max_concurrency:4}\00", section "llvm.metadata"

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
  call spir_func void @_Z3barv()
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
define spir_func void @_Z3barv() #3 {
entry:
  %s1 = alloca %struct.foo, align 4
  %0 = bitcast %struct.foo* %s1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 20, i8* %0) #4
  ; CHECK-LLVM: %[[FIELD1:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 0
  ; CHECK-LLVM: %[[CAST1:.*]] = bitcast{{.*}}%[[FIELD1]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST1]]{{.*}}[[STR1]]
  %f1 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 0
  %1 = bitcast i32* %f1 to i8*
  %2 = call i8* @llvm.ptr.annotation.p0i8(i8* %1, i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 2)
  %3 = bitcast i8* %2 to i32*
  store i32 0, i32* %3, align 4, !tbaa !9
  ; CHECK-LLVM: %[[FIELD2:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 1
  ; CHECK-LLVM: %[[CAST2:.*]] = bitcast{{.*}}%[[FIELD2]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST2]]{{.*}}[[STR2]]
  %f2 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 1
  %4 = bitcast i32* %f2 to i8*
  %5 = call i8* @llvm.ptr.annotation.p0i8(i8* %4, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 3)
  %6 = bitcast i8* %5 to i32*
  store i32 0, i32* %6, align 4, !tbaa !12
  ; CHECK-LLVM: %[[FIELD3:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 2
  ; CHECK-LLVM: %[[CAST3:.*]] = bitcast{{.*}}%[[FIELD3]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST3]]{{.*}}[[STR3]]
  %f3 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 2
  %7 = bitcast i32* %f3 to i8*
  %8 = call i8* @llvm.ptr.annotation.p0i8(i8* %7, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 4)
  %9 = bitcast i8* %8 to i32*
  store i32 0, i32* %9, align 4, !tbaa !13
  ; CHECK-LLVM: %[[FIELD4:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 3
  ; CHECK-LLVM: %[[CAST4:.*]] = bitcast{{.*}}%[[FIELD4]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST4]]{{.*}}[[STR4]]
  %f4 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 3
  %10 = bitcast i32* %f4 to i8*
  %11 = call i8* @llvm.ptr.annotation.p0i8(i8* %10, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 5)
  %12 = bitcast i8* %11 to i32*
  store i32 0, i32* %12, align 4, !tbaa !14
  ; CHECK-LLVM: %[[FIELD5:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 4
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[FIELD5]]{{.*}}[[STR5]]
  %f5 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 4
  %13 = call i8* @llvm.ptr.annotation.p0i8(i8* %f5, i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str.5, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 6)
  store i8 0, i8* %13, align 4, !tbaa !15
  %14 = bitcast %struct.foo* %s1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 20, i8* %14) #4
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
!10 = !{!"_ZTS3foo", !11, i64 0, !11, i64 4, !11, i64 8, !11, i64 12, !11, i64 16}
!11 = !{!"int", !7, i64 0}
!12 = !{!10, !11, i64 4}
!13 = !{!10, !11, i64 8}
!14 = !{!10, !11, i64 12}
!15 = !{!10, !11, i64 16}