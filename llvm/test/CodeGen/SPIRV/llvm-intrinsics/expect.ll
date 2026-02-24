; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-NOT: OpCapability ExpectAssumeKHR
; CHECK-SPIRV-NOT: OpExtension "SPV_KHR_expect_assume"
; CHECK-SPIRV:     OpFunction
; CHECK-SPIRV-NOT: %[[#]] = OpExpectKHR %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV:     %[[#RES1:]] = OpSConvert %[[#]] %[[#]]
; CHECK-SPIRV:     %[[#]] = OpINotEqual %[[#]] %[[#RES1]] %[[#]]

; CHECK-SPIRV:     OpFunction
; CHECK-SPIRV:     %[[#RES2:]] = OpSConvert %[[#]] %[[#]]
; CHECK-SPIRV-NOT: %[[#]] = OpExpectKHR %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV:     %[[#]] = OpINotEqual %[[#]] %[[#RES2]] %[[#]]

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

define spir_kernel void @_ZTSZ4mainE15kernel_function() {
entry:
  %0 = alloca %"class._ZTSZ4mainE3$_0.anon", align 1
  %1 = bitcast ptr %0 to ptr
  call void @llvm.lifetime.start.p0(i64 1, ptr %1)
  %2 = addrspacecast ptr %0 to ptr addrspace(4)
  call spir_func void @"_ZZ4mainENK3$_0clEv"(ptr addrspace(4) %2)
  %3 = bitcast ptr %0 to ptr
  call void @llvm.lifetime.end.p0(i64 1, ptr %3)
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

define internal spir_func void @"_ZZ4mainENK3$_0clEv"(ptr addrspace(4) %this) align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store ptr addrspace(4) %this, ptr %this.addr, align 8
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  %0 = bitcast ptr %a to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %0)
  %call = call spir_func i32 @_Z12expect_consti(i32 1)
  store i32 %call, ptr %a, align 4
  %1 = bitcast ptr %b to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %1)
  %call2 = call spir_func i32 @_Z10expect_funi(i32 2)
  store i32 %call2, ptr %b, align 4
  %2 = bitcast ptr %b to ptr
  call void @llvm.lifetime.end.p0(i64 4, ptr %2)
  %3 = bitcast ptr %a to ptr
  call void @llvm.lifetime.end.p0(i64 4, ptr %3)
  ret void
}

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

define spir_func i32 @_Z12expect_consti(i32 %x) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %conv = sext i32 %0 to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 1)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  %1 = load i32, ptr %x.addr, align 4
  store i32 %1, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %2 = load i32, ptr %retval, align 4
  ret i32 %2
}

define spir_func i32 @_Z10expect_funi(i32 %x) {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %conv = sext i32 %0 to i64
  %call = call spir_func i32 @_Z3foov()
  %conv1 = sext i32 %call to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 %conv1)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  %1 = load i32, ptr %x.addr, align 4
  store i32 %1, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %2 = load i32, ptr %retval, align 4
  ret i32 %2
}

declare i64 @llvm.expect.i64(i64, i64)

declare spir_func i32 @_Z3foov()
