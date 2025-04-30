; RUN: opt -S -passes=next-silicon-warn-unsupported-omp < %s 2>&1 | FileCheck %s

declare void @omp_set_lock(ptr noundef)
declare i32 @omp_get_num_procs()
declare i32 @omp_get_max_active_levels()
declare i32 @omp_get_num_places()
declare void @omp_display_affinity(ptr noundef)
declare i32 @omp_get_num_threads()

; CHECK: Warning: In function 'foo': OMP function call to unsupported function 'omp_set_lock'
; CHECK: Warning: In function 'foo': OMP function call to unsupported function 'omp_get_num_procs'
; CHECK: Warning: In function 'foo': OMP function call to unsupported function 'omp_get_max_active_levels'
; CHECK: Warning: In function 'foo': OMP function call to unsupported function 'omp_get_num_places'
; CHECK: Warning: In function 'foo': OMP function call to unsupported function 'omp_display_affinity'
; CHECK: Warning: In function 'foo': OMP function call to non-conforming function 'omp_get_num_threads'
define void @foo() {
  %1 = alloca i32
  call void @omp_set_lock(ptr %1)
  %2 = call i32 @omp_get_num_procs()
  %3 = call i32 @omp_get_max_active_levels()
  %4 = call i32 @omp_get_num_places()
  call void @omp_display_affinity(ptr %1)
  %5 = call i32 @omp_get_num_threads()
  ret void
}