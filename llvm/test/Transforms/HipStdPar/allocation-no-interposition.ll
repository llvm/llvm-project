; RUN: opt < %s -passes=hipstdpar-interpose-alloc -S 2>&1 | FileCheck %s

; CHECK: warning: {{.*}} aligned_alloc {{.*}} cannot be interposed, missing: __hipstdpar_aligned_alloc. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} free {{.*}} cannot be interposed, missing: __hipstdpar_free. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} calloc {{.*}} cannot be interposed, missing: __hipstdpar_calloc. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} malloc {{.*}} cannot be interposed, missing: __hipstdpar_malloc. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} memalign {{.*}} cannot be interposed, missing: __hipstdpar_aligned_alloc. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} posix_memalign {{.*}} cannot be interposed, missing: __hipstdpar_posix_aligned_alloc. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} realloc {{.*}} cannot be interposed, missing: __hipstdpar_realloc. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} reallocarray {{.*}} cannot be interposed, missing: __hipstdpar_realloc_array. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _Znwm {{.*}} cannot be interposed, missing: __hipstdpar_operator_new. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZdlPv {{.*}} cannot be interposed, missing: __hipstdpar_operator_delete. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZnwmSt11align_val_t {{.*}} cannot be interposed, missing: __hipstdpar_operator_new_aligned. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZdlPvSt11align_val_t {{.*}} cannot be interposed, missing: __hipstdpar_operator_delete_aligned. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZnwmRKSt9nothrow_t {{.*}} cannot be interposed, missing: __hipstdpar_operator_new_nothrow. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZnwmSt11align_val_tRKSt9nothrow_t {{.*}} cannot be interposed, missing: __hipstdpar_operator_new_aligned_nothrow. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _Znam {{.*}} cannot be interposed, missing: __hipstdpar_operator_new. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZdaPv {{.*}} cannot be interposed, missing: __hipstdpar_operator_delete. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZnamSt11align_val_t {{.*}} cannot be interposed, missing: __hipstdpar_operator_new_aligned. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZdaPvSt11align_val_t {{.*}} cannot be interposed, missing: __hipstdpar_operator_delete_aligned. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZnamRKSt9nothrow_t {{.*}} cannot be interposed, missing: __hipstdpar_operator_new_nothrow. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} _ZnamSt11align_val_tRKSt9nothrow_t {{.*}} cannot be interposed, missing: __hipstdpar_operator_new_aligned_nothrow. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} __libc_calloc {{.*}} cannot be interposed, missing: __hipstdpar_calloc. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} __libc_free {{.*}} cannot be interposed, missing: __hipstdpar_free. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} __libc_malloc {{.*}} cannot be interposed, missing: __hipstdpar_malloc. Tried to run the allocation interposition pass without the replacement functions available.
; CHECK: warning: {{.*}} __libc_memalign {{.*}} cannot be interposed, missing: __hipstdpar_aligned_alloc. Tried to run the allocation interposition pass without the replacement functions available.

%"struct.std::nothrow_t" = type { i8 }

@_ZSt7nothrow = external global %"struct.std::nothrow_t", align 1

define dso_local noundef i32 @allocs() {
  %1 = call noalias align 8 ptr @aligned_alloc(i64 noundef 8, i64 noundef 42)
  call void @free(ptr noundef %1)

  %2 = call noalias ptr @calloc(i64 noundef 1, i64 noundef 42)
  call void @free(ptr noundef %2)

  %3 = call noalias ptr @malloc(i64 noundef 42)
  call void @free(ptr noundef %3)

  %4 = call noalias align 8 ptr @memalign(i64 noundef 8, i64 noundef 42)
  call void @free(ptr noundef %4)

  %tmp = alloca ptr, align 8
  %5 = call i32 @posix_memalign(ptr noundef %tmp, i64 noundef 8, i64 noundef 42)
  call void @free(ptr noundef %tmp)

  %6 = call noalias ptr @malloc(i64 noundef 42)
  %7 = call ptr @realloc(ptr noundef %6, i64 noundef 42)
  call void @free(ptr noundef %7)

  %8 = call noalias ptr @calloc(i64 noundef 1, i64 noundef 42)
  %9 = call ptr @reallocarray(ptr noundef %8, i64 noundef 1, i64 noundef 42)
  call void @free(ptr noundef %9)

  %10 = call noalias noundef nonnull ptr @_Znwm(i64 noundef 1)
  call void @_ZdlPv(ptr noundef %10)

  %11 = call noalias noundef nonnull align 8 ptr @_ZnwmSt11align_val_t(i64 noundef 1, i64 noundef 8)
  call void @_ZdlPvSt11align_val_t(ptr noundef %11, i64 noundef 8)

  %12 = call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 1, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  call void @_ZdlPv(ptr noundef %12)

  %13 = call noalias noundef align 8 ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 noundef 1, i64 noundef 8, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  call void @_ZdlPvSt11align_val_t(ptr noundef %13, i64 noundef 8)

  %14 = call noalias noundef nonnull ptr @_Znam(i64 noundef 42)
  call void @_ZdaPv(ptr noundef %14)

  %15 = call noalias noundef nonnull align 8 ptr @_ZnamSt11align_val_t(i64 noundef 42, i64 noundef 8)
  call void @_ZdaPvSt11align_val_t(ptr noundef %15, i64 noundef 8)

  %16 = call noalias noundef ptr @_ZnamRKSt9nothrow_t(i64 noundef 42, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  call void @_ZdaPv(ptr noundef %16)

  %17 = call noalias noundef align 8 ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 noundef 42, i64 noundef 8, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  call void @_ZdaPvSt11align_val_t(ptr noundef %17, i64 noundef 8)

  %18 = call ptr @calloc(i64 noundef 1, i64 noundef 42)
  call void @free(ptr noundef %18)

  %19 = call ptr @malloc(i64 noundef 42)
  call void @free(ptr noundef %19)

  %20 = call noalias noundef nonnull ptr @_Znwm(i64 noundef 42)
  call void @_ZdlPv(ptr noundef %20)

  %21 = call noalias noundef nonnull align 8 ptr @_ZnwmSt11align_val_t(i64 noundef 42, i64 noundef 8)
  call void @_ZdlPvSt11align_val_t(ptr noundef %21, i64 noundef 8)

  %22 = call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef 42, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  call void @_ZdlPv(ptr noundef %22)

  %23 = call noalias noundef align 8 ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 noundef 42, i64 noundef 8, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow)
  call void @_ZdlPvSt11align_val_t(ptr noundef %23, i64 noundef 8)

  %24 = call ptr @malloc(i64 noundef 42)
  %25 = call ptr @realloc(ptr noundef %24, i64 noundef 41)
  call void @free(ptr noundef %25)

  %26 = call ptr @__libc_calloc(i64 noundef 1, i64 noundef 42)
  call void @__libc_free(ptr noundef %26)

  %27 = call ptr @__libc_malloc(i64 noundef 42)
  call void @__libc_free(ptr noundef %27)

  %28 = call ptr @__libc_memalign(i64 noundef 8, i64 noundef 42)
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