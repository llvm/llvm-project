; RUN: opt -S -passes=hipstdpar-interpose-alloc %s | FileCheck %s

%"struct.std::nothrow_t" = type { i8 }

@_ZSt7nothrow = external global %"struct.std::nothrow_t", align 1

declare ptr @__hipstdpar_aligned_alloc(i64, i64)

declare ptr @__hipstdpar_malloc(i64)

declare ptr @__hipstdpar_calloc(i64, i64)

declare i32 @__hipstdpar_posix_aligned_alloc(ptr, i64, i64)

declare void @__hipstdpar_hidden_free(ptr)

declare ptr @__hipstdpar_realloc(ptr, i64)

declare ptr @__hipstdpar_realloc_array(ptr, i64, i64)

declare void @__hipstdpar_free(ptr)

declare ptr @__hipstdpar_operator_new_aligned(i64, i64)

declare ptr @__hipstdpar_operator_new(i64)

declare ptr @__hipstdpar_operator_new_nothrow(i64, %"struct.std::nothrow_t")

declare ptr @__hipstdpar_operator_new_aligned_nothrow(i64, i64, %"struct.std::nothrow_t")

declare void @__hipstdpar_operator_delete_aligned_sized(ptr, i64, i64)

declare void @__hipstdpar_operator_delete(ptr)

declare void @__hipstdpar_operator_delete_aligned(ptr, i64)

declare void @__hipstdpar_operator_delete_sized(ptr, i64)

define dso_local noundef i32 @allocs() {
  ; CHECK: %1 = call noalias align 8 ptr @__hipstdpar_aligned_alloc(i64 noundef 8, i64 noundef 42)
  %1 = call noalias align 8 ptr @aligned_alloc(i64 noundef 8, i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %1)
  call void @free(ptr noundef %1)

  ; CHECK: %2 = call noalias ptr @__hipstdpar_calloc(i64 noundef 1, i64 noundef 42)
  %2 = call noalias ptr @calloc(i64 noundef 1, i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %2)
  call void @free(ptr noundef %2)

  ; CHECK: %3 = call noalias ptr @__hipstdpar_malloc(i64 noundef 42)
  %3 = call noalias ptr @malloc(i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %3)
  call void @free(ptr noundef %3)

  ; CHECK: %4 = call noalias align 8 ptr @__hipstdpar_aligned_alloc(i64 noundef 8, i64 noundef 42)
  %4 = call noalias align 8 ptr @memalign(i64 noundef 8, i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %4)
  call void @free(ptr noundef %4)

  %tmp = alloca ptr, align 8
  ; CHECK: %5 = call i32 @__hipstdpar_posix_aligned_alloc(ptr noundef %tmp, i64 noundef 8, i64 noundef 42)
  %5 = call i32 @posix_memalign(ptr noundef %tmp, i64 noundef 8, i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %tmp)
  call void @free(ptr noundef %tmp)

  ; CHECK: %6 = call noalias ptr @__hipstdpar_malloc(i64 noundef 42)
  %6 = call noalias ptr @malloc(i64 noundef 42)
  ; CHECK: %7 = call ptr @__hipstdpar_realloc(ptr noundef %6, i64 noundef 42)
  %7 = call ptr @realloc(ptr noundef %6, i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %7)
  call void @free(ptr noundef %7)

  ; CHECK: %8 = call noalias ptr @__hipstdpar_calloc(i64 noundef 1, i64 noundef 42)
  %8 = call noalias ptr @calloc(i64 noundef 1, i64 noundef 42)
  ; CHECK: %9 = call ptr @__hipstdpar_realloc_array(ptr noundef %8, i64 noundef 1, i64 noundef 42)
  %9 = call ptr @reallocarray(ptr noundef %8, i64 noundef 1, i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %9)
  call void @free(ptr noundef %9)

  ; CHECK: %10 = call noalias noundef nonnull ptr @__hipstdpar_operator_new(i64 noundef 1)
  %10 = call noalias noundef nonnull ptr @_Znwm(i64 noundef 1)
  ; CHECK: call void @__hipstdpar_operator_delete(ptr noundef %10)
  call void @_ZdlPv(ptr noundef %10)

  ; CHECK: %11 = call noalias noundef nonnull align 8 ptr @__hipstdpar_operator_new_aligned(i64 noundef 1, i64 noundef 8)
  %11 = call noalias noundef nonnull align 8 ptr @_ZnwmSt11align_val_t(i64 noundef 1, i64 noundef 8)
  ; CHECK: call void @__hipstdpar_operator_delete_aligned(ptr noundef %11, i64 noundef 8)
  call void @_ZdlPvSt11align_val_t(ptr noundef %11, i64 noundef 8)

  ; CHECK: %12 = call noalias noundef ptr @__hipstdpar_operator_new_nothrow(i64 noundef 1, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  %12 = call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 1, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  ; CHECK: call void @__hipstdpar_operator_delete(ptr noundef %12)
  call void @_ZdlPv(ptr noundef %12)

  ; CHECK: %13 = call noalias noundef align 8 ptr @__hipstdpar_operator_new_aligned_nothrow(i64 noundef 1, i64 noundef 8, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  %13 = call noalias noundef align 8 ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 noundef 1, i64 noundef 8, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  ; CHECK: call void @__hipstdpar_operator_delete_aligned(ptr noundef %13, i64 noundef 8)
  call void @_ZdlPvSt11align_val_t(ptr noundef %13, i64 noundef 8)

  ; CHECK: %14 = call noalias noundef nonnull ptr @__hipstdpar_operator_new(i64 noundef 42)
  %14 = call noalias noundef nonnull ptr @_Znam(i64 noundef 42)
  ; CHECK: call void @__hipstdpar_operator_delete(ptr noundef %14)
  call void @_ZdaPv(ptr noundef %14)

  ; CHECK: %15 = call noalias noundef nonnull align 8 ptr @__hipstdpar_operator_new_aligned(i64 noundef 42, i64 noundef 8)
  %15 = call noalias noundef nonnull align 8 ptr @_ZnamSt11align_val_t(i64 noundef 42, i64 noundef 8)
  ; CHECK: call void @__hipstdpar_operator_delete_aligned(ptr noundef %15, i64 noundef 8)
  call void @_ZdaPvSt11align_val_t(ptr noundef %15, i64 noundef 8)

  ; CHECK:  %16 = call noalias noundef ptr @__hipstdpar_operator_new_nothrow(i64 noundef 42, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  %16 = call noalias noundef ptr @_ZnamRKSt9nothrow_t(i64 noundef 42, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  ; CHECK: call void @__hipstdpar_operator_delete(ptr noundef %16)
  call void @_ZdaPv(ptr noundef %16)

  ; CHECK: %17 = call noalias noundef align 8 ptr @__hipstdpar_operator_new_aligned_nothrow(i64 noundef 42, i64 noundef 8, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  %17 = call noalias noundef align 8 ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 noundef 42, i64 noundef 8, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  ; CHECK: call void @__hipstdpar_operator_delete_aligned(ptr noundef %17, i64 noundef 8)
  call void @_ZdaPvSt11align_val_t(ptr noundef %17, i64 noundef 8)

  ; CHECK:  %18 = call ptr @__hipstdpar_calloc(i64 noundef 1, i64 noundef 42)
  %18 = call ptr @calloc(i64 noundef 1, i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %18)
  call void @free(ptr noundef %18)

  ; CHECK: %19 = call ptr @__hipstdpar_malloc(i64 noundef 42)
  %19 = call ptr @malloc(i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %19)
  call void @free(ptr noundef %19)

  ; CHECK: %20 = call noalias noundef nonnull ptr @__hipstdpar_operator_new(i64 noundef 42)
  %20 = call noalias noundef nonnull ptr @_Znwm(i64 noundef 42)
  ; CHECK: call void @__hipstdpar_operator_delete(ptr noundef %20)
  call void @_ZdlPv(ptr noundef %20)

  ; CHECK:  %21 = call noalias noundef nonnull align 8 ptr @__hipstdpar_operator_new_aligned(i64 noundef 42, i64 noundef 8)
  %21 = call noalias noundef nonnull align 8 ptr @_ZnwmSt11align_val_t(i64 noundef 42, i64 noundef 8)
  ; CHECK: call void @__hipstdpar_operator_delete_aligned(ptr noundef %21, i64 noundef 8)
  call void @_ZdlPvSt11align_val_t(ptr noundef %21, i64 noundef 8)

  ; CHECK: %22 = call noalias noundef ptr @__hipstdpar_operator_new_nothrow(i64 noundef 42, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  %22 = call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 42, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  ; CHECK: call void @__hipstdpar_operator_delete(ptr noundef %22)
  call void @_ZdlPv(ptr noundef %22)

  ; CHECK:  %23 = call noalias noundef align 8 ptr @__hipstdpar_operator_new_aligned_nothrow(i64 noundef 42, i64 noundef 8, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  %23 = call noalias noundef align 8 ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 noundef 42, i64 noundef 8, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  ; CHECK: call void @__hipstdpar_operator_delete_aligned(ptr noundef %23, i64 noundef 8)
  call void @_ZdlPvSt11align_val_t(ptr noundef %23, i64 noundef 8)

  ; CHECK: %24 = call ptr @__hipstdpar_malloc(i64 noundef 42)
  %24 = call ptr @malloc(i64 noundef 42)
  ; CHECK: %25 = call ptr @__hipstdpar_realloc(ptr noundef %24, i64 noundef 41)
  %25 = call ptr @realloc(ptr noundef %24, i64 noundef 41)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %25)
  call void @free(ptr noundef %25)

  ; CHECK: %26 = call ptr @__hipstdpar_calloc(i64 noundef 1, i64 noundef 42)
  %26 = call ptr @__libc_calloc(i64 noundef 1, i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %26)
  call void @__libc_free(ptr noundef %26)

  ; CHECK: %27 = call ptr @__hipstdpar_malloc(i64 noundef 42)
  %27 = call ptr @__libc_malloc(i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %27)
  call void @__libc_free(ptr noundef %27)

  ; CHECK: %28 = call ptr @__hipstdpar_aligned_alloc(i64 noundef 8, i64 noundef 42)
  %28 = call ptr @__libc_memalign(i64 noundef 8, i64 noundef 42)
  ; CHECK: call void @__hipstdpar_free(ptr noundef %28)
  call void @__libc_free(ptr noundef %28)

  ret i32 0
}

declare noalias ptr @aligned_alloc(i64 noundef, i64 noundef)

declare void @free(ptr noundef)

declare noalias ptr @calloc(i64 noundef, i64 noundef)

declare noalias ptr @malloc(i64 noundef)

declare noalias ptr @memalign(i64 noundef, i64 noundef)

declare i32 @posix_memalign(ptr noundef, i64 noundef, i64 noundef)

declare ptr @realloc(ptr noundef, i64 noundef)

declare ptr @reallocarray(ptr noundef, i64 noundef, i64 noundef)

declare noundef nonnull ptr @_Znwm(i64 noundef)

declare void @_ZdlPv(ptr noundef)

declare noalias noundef nonnull ptr @_ZnwmSt11align_val_t(i64 noundef, i64 noundef)

declare void @_ZdlPvSt11align_val_t(ptr noundef, i64 noundef)

declare noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef, ptr noundef nonnull align 1 dereferenceable(1))

declare noalias noundef ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 noundef, i64 noundef, ptr noundef nonnull align 1 dereferenceable(1))

declare noundef nonnull ptr @_Znam(i64 noundef)

declare void @_ZdaPv(ptr noundef)

declare noalias noundef nonnull ptr @_ZnamSt11align_val_t(i64 noundef, i64 noundef)

declare void @_ZdaPvSt11align_val_t(ptr noundef, i64 noundef)

declare noalias noundef ptr @_ZnamRKSt9nothrow_t(i64 noundef, ptr noundef nonnull align 1 dereferenceable(1))

declare noalias noundef ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 noundef, i64 noundef, ptr noundef nonnull align 1 dereferenceable(1))

declare ptr @__libc_calloc(i64 noundef, i64 noundef)

declare void @__libc_free(ptr noundef)

declare ptr @__libc_malloc(i64 noundef)

declare ptr @__libc_memalign(i64 noundef, i64 noundef)