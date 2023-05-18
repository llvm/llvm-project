; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=ifuncs --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -implicit-check-not=ifunc_remove --check-prefixes=CHECK-FINAL --input-file=%t %s

target datalayout = "P1-G2"

define void @existing_ctor() addrspace(1) {
  ret void
}

; Currently don't support expanding these uses.
; CHECK-FINAL: @constant_init_user_addrspace1 = global ptr addrspace(1) @ifunc_remove_used_in_constantinit_as1
; CHECK-FINAL: @constant_init_user_addrspace1_cast = global ptr addrspace(1) addrspacecast (ptr @ifunc_remove_used_in_constantinit_as1_cast to ptr addrspace(1))
; CHECK-FINAL: @constant_init_user_addrspace0_cast = global ptr addrspacecast (ptr addrspace(1) @ifunc_remove_used_in_constantinit_as0_cast to ptr)


; CHECK-FINAL: [[TABLE:@[0-9]+]] = internal addrspace(2) global [6 x ptr addrspace(1)] poison, align 8

; CHECK-FINAL: @llvm.global_ctors = appending addrspace(2) global [2 x { i32, ptr addrspace(1), ptr }] [{ i32, ptr addrspace(1), ptr } { i32 0, ptr addrspace(1) @existing_ctor, ptr null }, { i32, ptr addrspace(1), ptr } { i32 10, ptr addrspace(1) [[TABLE_CTOR:@[0-9]+]], ptr null }]
@llvm.global_ctors = appending global [1 x { i32, ptr addrspace(1), ptr }] [{ i32, ptr addrspace(1), ptr } { i32 0, ptr addrspace(1) @existing_ctor, ptr null }]



; CHECK-FINAL: @ifunc_remove_used_in_constantinit_as1 = ifunc i32 (double), ptr addrspace(1) @resolver1_in_1
; CHECK-FINAL: @ifunc_remove_used_in_constantinit_as1_cast = ifunc i32 (double), ptr @resolver1_in_0
; CHECK-FINAL: @ifunc_remove_used_in_constantinit_as0_cast = ifunc i32 (double), ptr addrspace(1) @resolver0_in_1

@ifunc_remove_used_in_constantinit_as1 = ifunc i32 (double), ptr addrspace(1) @resolver1_in_1
@constant_init_user_addrspace1 = global ptr addrspace(1) @ifunc_remove_used_in_constantinit_as1

@ifunc_remove_used_in_constantinit_as1_cast = ifunc i32 (double), ptr @resolver1_in_0
@constant_init_user_addrspace1_cast = global ptr addrspace(1) addrspacecast (ptr @ifunc_remove_used_in_constantinit_as1_cast to ptr addrspace(1))

@ifunc_remove_used_in_constantinit_as0_cast = ifunc i32 (double), ptr addrspace(1) @resolver0_in_1
@constant_init_user_addrspace0_cast = global ptr addrspacecast (ptr addrspace(1) @ifunc_remove_used_in_constantinit_as0_cast to ptr)


; CHECK-INTERESTINGNESS: @ifunc_keep_as1_resolver_in_0
; CHECK-FINAL: @ifunc_keep_as1_resolver_in_0 = ifunc void (), ptr @resolver1_in_0
@ifunc_keep_as1_resolver_in_0 = ifunc void (), ptr @resolver1_in_0
@ifunc_remove_as1_resolver_in_0 = ifunc void (), ptr @resolver1_in_0

; CHECK-INTERESTINGNESS: @ifunc_keep_as1_resolver_in_1
; CHECK-FINAL: @ifunc_keep_as1_resolver_in_1 = ifunc void (), ptr addrspace(1) @resolver1_in_1
@ifunc_keep_as1_resolver_in_1 = ifunc void (), ptr addrspace(1) @resolver1_in_1
@ifunc_remove_as1_resolver_in_1 = ifunc void (), ptr addrspace(1) @resolver1_in_1

; CHECK-INTERESTINGNESS: @ifunc_keep_as1_resolver_casted_in_1
; CHECK-FINAL: @ifunc_keep_as1_resolver_casted_in_1 = ifunc void (), addrspacecast (ptr @resolver1_in_0 to ptr addrspace(1))
@ifunc_keep_as1_resolver_casted_in_1 = ifunc void (), ptr addrspace(1) addrspacecast (ptr @resolver1_in_0 to ptr addrspace(1))
@ifunc_remove_as1_resolver_casted_in_1 = ifunc void (), ptr addrspace(1) addrspacecast (ptr @resolver1_in_0 to ptr addrspace(1))


define ptr addrspace(1) @resolver1_in_0() addrspace(0) {
  ret ptr addrspace(1) inttoptr (i64 123 to ptr addrspace(1))
}

define ptr addrspace(1) @resolver1_in_1() addrspace(1) {
  ret ptr addrspace(1) inttoptr (i64 456 to ptr addrspace(1))
}

define ptr addrspace(0) @resolver0_in_1() addrspace(1) {
  ret ptr addrspace(0) inttoptr (i64 789 to ptr addrspace(0))
}

define void @call_removed() addrspace(0) {
  ; CHECK-FINAL-LABEL: @call_removed(
  ; CHECK-FINAL-NEXT: %1 = load ptr addrspace(1), ptr addrspace(2) getelementptr inbounds ([6 x ptr addrspace(1)], ptr addrspace(2) [[TABLE]], i32 0, i32 3), align 8
  ; CHECK-FINAL-NEXT: %2 = addrspacecast ptr addrspace(1) %1 to ptr
  ; CHECK-FINAL-NEXT: call addrspace(0) void %2()
  ; CHECK-FINAL-NEXT: %3 = load ptr addrspace(1), ptr addrspace(2) getelementptr inbounds ([6 x ptr addrspace(1)], ptr addrspace(2) [[TABLE]], i32 0, i32 4), align 8
  ; CHECK-FINAL-NEXT: call addrspace(1) void %3()
  ; CHECK-FINAL-NEXT: %4 = load ptr addrspace(1), ptr addrspace(2) getelementptr inbounds ([6 x ptr addrspace(1)], ptr addrspace(2) [[TABLE]], i32 0, i32 5), align 8
  ; CHECK-FINAL-NEXT: call addrspace(1) void %4()
  ; CHECK-FINAL-NEXT: ret void
  call addrspace(0) void @ifunc_remove_as1_resolver_in_0()
  call addrspace(1) void @ifunc_remove_as1_resolver_in_1()
  call addrspace(1) void @ifunc_remove_as1_resolver_casted_in_1()
  ret void
}

define void @load_removed() addrspace(0) {
  ; CHECK-FINAL-LABEL: define void @load_removed(
  ; CHECK-FINAL-NEXT: %load0 = load volatile ptr addrspace(1), ptr @constant_init_user_addrspace1, align 8
  ; CHECK-FINAL-NEXT: %load1 = load volatile ptr addrspace(1), ptr @constant_init_user_addrspace1_cast, align 8
  ; CHECK-FINAL-NEXT: %load2 = load volatile ptr, ptr @constant_init_user_addrspace0_cast, align 8
  ; CHECK-FINAL-NEXT: ret void
  %load0 = load volatile ptr addrspace(1), ptr @constant_init_user_addrspace1
  %load1 = load volatile ptr addrspace(1), ptr @constant_init_user_addrspace1_cast
  %load2 = load volatile ptr, ptr @constant_init_user_addrspace0_cast
  ret void
}

; CHECK-FINAL: define internal void [[TABLE_CTOR]]() addrspace(1) {
; CHECK-FINAL-NEXT: %1 = call addrspace(1) ptr addrspace(1) @resolver1_in_1()
; CHECK-FINAL-NEXT: store ptr addrspace(1) %1, ptr addrspace(2) [[TABLE]], align 8
; CHECK-FINAL-NEXT: %2 = call addrspace(0) ptr addrspace(1) @resolver1_in_0()
; CHECK-FINAL-NEXT: store ptr addrspace(1) %2, ptr addrspace(2) getelementptr inbounds ([6 x ptr addrspace(1)], ptr addrspace(2) [[TABLE]], i32 0, i32 1), align 8
; CHECK-FINAL-NEXT: %3 = call addrspace(1) ptr @resolver0_in_1()
; CHECK-FINAL-NEXT: %4 = addrspacecast ptr %3 to ptr addrspace(1)
; CHECK-FINAL-NEXT: store ptr addrspace(1) %4, ptr addrspace(2) getelementptr inbounds ([6 x ptr addrspace(1)], ptr addrspace(2) [[TABLE]], i32 0, i32 2), align 8
; CHECK-FINAL-NEXT: %5 = call addrspace(0) ptr addrspace(1) @resolver1_in_0()
; CHECK-FINAL-NEXT: store ptr addrspace(1) %5, ptr addrspace(2) getelementptr inbounds ([6 x ptr addrspace(1)], ptr addrspace(2) [[TABLE]], i32 0, i32 3), align 8
; CHECK-FINAL-NEXT: %6 = call addrspace(1) ptr addrspace(1) @resolver1_in_1()
; CHECK-FINAL-NEXT: store ptr addrspace(1) %6, ptr addrspace(2) getelementptr inbounds ([6 x ptr addrspace(1)], ptr addrspace(2) [[TABLE]], i32 0, i32 4), align 8
; CHECK-FINAL-NEXT: %7 = call addrspace(0) ptr addrspace(1) @resolver1_in_0()
; CHECK-FINAL-NEXT: store ptr addrspace(1) %7, ptr addrspace(2) getelementptr inbounds ([6 x ptr addrspace(1)], ptr addrspace(2) [[TABLE]], i32 0, i32 5), align 8
; CHECK-FINAL-NEXT: ret void
