; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3 %s -print-after-all -o - 2>&1 | FileCheck %s

; CHECK: *** IR Dump After Fixup address space (spirv-fix-address-space) ***

%struct.S = type { i32 }

@object = internal global %struct.S zeroinitializer, align 4
; CHECK: @object = internal addrspace(10) global %struct.S zeroinitializer
@input = internal global i32 0, align 4
; CHECK: @input = internal addrspace(10) global i32 0
@output = internal global i32 0, align 4
; CHECK: @output = internal addrspace(10) global i32 0

define linkonce_odr spir_func void @S_set(ptr noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %param) #0 align 2 {
; CHECK: define linkonce_odr spir_func void @S_set(ptr addrspace(10) noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %param) #0 align 2 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %value = getelementptr inbounds nuw %struct.S, ptr %this, i32 0, i32 0
; CHECK: %value1 = getelementptr inbounds nuw %struct.S, ptr addrspace(10) %this, i32 0, i32 0
  store i32 %param, ptr %value, align 4
; CHECK: store i32 %param, ptr addrspace(10) %value1, align 4
  ret void
}

define linkonce_odr spir_func noundef i32 @S_get(ptr noundef nonnull align 4 dereferenceable(4) %this) #0 align 2 {
; CHECK: define linkonce_odr spir_func noundef i32 @S_get(ptr addrspace(10) noundef nonnull align 4 dereferenceable(4) %this) #0 align 2 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %value = getelementptr inbounds nuw %struct.S, ptr %this, i32 0, i32 0
; CHECK: %value1 = getelementptr inbounds nuw %struct.S, ptr addrspace(10) %this, i32 0, i32 0
  %1 = load i32, ptr %value, align 4
; CHECK: %1 = load i32, ptr addrspace(10) %value1, align 4
  ret i32 %1
}

define void @main() #1 {
; CHECK: define void @main() #1 {
entry:
; CHECK: entry:
  %0 = call token @llvm.experimental.convergence.entry()

  %1 = load i32, ptr @input, align 4
; %1 = load i32, ptr addrspace(10) @input, align 4
  call spir_func void @S_set(ptr noundef nonnull align 4 dereferenceable(4) @object, i32 noundef %1) #0 [ "convergencectrl"(token %0) ]
; call spir_func void @S_set(ptr addrspace(10) noundef nonnull align 4 dereferenceable(4) @object, i32 noundef %1) #0 [ "convergencectrl"(token %0) ]

  %call1 = call spir_func noundef i32 @S_get(ptr noundef nonnull align 4 dereferenceable(4) @object) #0 [ "convergencectrl"(token %0) ]
; %call1 = call spir_func noundef i32 @S_get(ptr addrspace(10) noundef nonnull align 4 dereferenceable(4) @object) #0 [ "convergencectrl"(token %0) ]

  store i32 %call1, ptr @output, align 4
; store i32 %call1, ptr addrspace(10) @output, align 4
  ret void
; ret void
}

declare token @llvm.experimental.convergence.entry() #2

attributes #0 = { convergent alwaysinline }
attributes #1 = { convergent noinline norecurse }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
