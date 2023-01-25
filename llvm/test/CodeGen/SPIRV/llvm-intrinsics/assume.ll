; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-NOT: OpCapability ExpectAssumeKHR
; CHECK-SPIRV-NOT: OpExtension "SPV_KHR_expect_assume"
; CHECK-SPIRV:     OpName %[[#COMPARE:]] "cmp"
; CHECK-SPIRV:     %[[#COMPARE]] = OpINotEqual %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV-NOT: OpAssumeTrueKHR %[[#COMPARE]]

%class.anon = type { i8 }

define spir_func void @_Z3fooi(i32 %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %cmp = icmp ne i32 %0, 0
  call void @llvm.assume(i1 %cmp)
  ret void
}

declare void @llvm.assume(i1)

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %agg.tmp = alloca %class.anon, align 1
  store i32 0, i32* %retval, align 4
  call spir_func void @"_Z18kernel_single_taskIZ4mainE11fake_kernelZ4mainE3$_0EvT0_"(%class.anon* byval(%class.anon) align 1 %agg.tmp)
  ret i32 0
}

define internal spir_func void @"_Z18kernel_single_taskIZ4mainE11fake_kernelZ4mainE3$_0EvT0_"(%class.anon* byval(%class.anon) align 1 %kernelFunc) {
entry:
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %kernelFunc)
  ret void
}

define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %this) align 2 {
entry:
  %this.addr = alloca %class.anon*, align 8
  %a = alloca i32, align 4
  store %class.anon* %this, %class.anon** %this.addr, align 8
  %this1 = load %class.anon*, %class.anon** %this.addr, align 8
  %0 = bitcast i32* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0)
  store i32 1, i32* %a, align 4
  %1 = load i32, i32* %a, align 4
  call spir_func void @_Z3fooi(i32 %1)
  %2 = bitcast i32* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
