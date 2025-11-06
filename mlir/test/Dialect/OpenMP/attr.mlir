// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: module attributes {omp.flags = #omp.flags<>} {
module attributes {omp.flags = #omp.flags<debug_kind = 0, assume_teams_oversubscription = false, assume_threads_oversubscription = false, assume_no_thread_state = false, assume_no_nested_parallelism = false, openmp_device_version = 50>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 20>} {
module attributes {omp.flags = #omp.flags<debug_kind = 20, assume_teams_oversubscription = false, assume_threads_oversubscription = false, assume_no_thread_state = false, assume_no_nested_parallelism = false>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 100, assume_teams_oversubscription = true>} {
module attributes {omp.flags = #omp.flags<debug_kind = 100, assume_teams_oversubscription = true, assume_threads_oversubscription = false, assume_no_thread_state = false, assume_no_nested_parallelism = false>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 200, assume_teams_oversubscription = true, assume_threads_oversubscription = true>} {
module attributes {omp.flags = #omp.flags<debug_kind = 200, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = false, assume_no_nested_parallelism = false>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 300, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true>} {
module attributes {omp.flags = #omp.flags<debug_kind = 300, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = false>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 400, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = true>} {
module attributes {omp.flags = #omp.flags<debug_kind = 400, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = true>} {}

// CHECK: module attributes {omp.flags = #omp.flags<>} {
module attributes {omp.flags = #omp.flags<>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true>} {
module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true>} {
module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true>} {
module attributes {omp.flags = #omp.flags<assume_no_thread_state = true, assume_teams_oversubscription = true>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 20, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<debug_kind = 20, assume_teams_oversubscription = false, assume_threads_oversubscription = false, assume_no_thread_state = false, assume_no_nested_parallelism = false, openmp_device_version = 51>} {}

//: module attributes {omp.flags = #omp.flags<debug_kind = 100, assume_teams_oversubscription = true, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<debug_kind = 100, assume_teams_oversubscription = true, assume_threads_oversubscription = false, assume_no_thread_state = false, assume_no_nested_parallelism = false, openmp_device_version = 51>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 200, assume_teams_oversubscription = true, assume_threads_oversubscription = true, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<debug_kind = 200, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = false, assume_no_nested_parallelism = false, openmp_device_version = 51>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 300, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<debug_kind = 300, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = false, openmp_device_version = 51>} {}

// CHECK: module attributes {omp.flags = #omp.flags<debug_kind = 400, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = true, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<debug_kind = 400, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = true, openmp_device_version = 51>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, openmp_device_version = 51>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true, openmp_device_version = 51>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<assume_no_thread_state = true, assume_teams_oversubscription = true, openmp_device_version = 51>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, assume_no_thread_state = true, no_gpu_lib = true, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<assume_no_thread_state = true, assume_teams_oversubscription = true, no_gpu_lib = true, openmp_device_version = 51>} {}

// CHECK: module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, openmp_device_version = 51>} {
module attributes {omp.flags = #omp.flags<assume_teams_oversubscription = true, no_gpu_lib = false, openmp_device_version = 51>} {}

// CHECK: module attributes {omp.version = #omp.version<version = 51>} {
module attributes {omp.version = #omp.version<version = 51>} {}

// ----

// CHECK-LABEL: func @omp_decl_tar_host_to
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} {
func.func @omp_decl_tar_host_to() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_host_link
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (link)>} {
func.func @omp_decl_tar_host_link() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (link)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_host_enter
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>} {
func.func @omp_decl_tar_host_enter() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_nohost_to
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} {
func.func @omp_decl_tar_nohost_to() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_nohost_link
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (link)>} {
func.func @omp_decl_tar_nohost_link() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (link)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_nohost_enter
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>} {
func.func @omp_decl_tar_nohost_enter() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_any_to
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
func.func @omp_decl_tar_any_to() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_any_link
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} {
func.func @omp_decl_tar_any_link() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_any_enter
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
func.func @omp_decl_tar_any_enter() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
  return
}

// CHECK-LABEL: global external @omp_decl_tar_data_host_to
// CHECK-SAME: {{.*}} {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>}
llvm.mlir.global external @omp_decl_tar_data_host_to() {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: global external @omp_decl_tar_data_host_link
// CHECK-SAME: {{.*}} {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (link)>}
llvm.mlir.global external @omp_decl_tar_data_host_link() {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (link)>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: global external @omp_decl_tar_data_host_enter
// CHECK-SAME: {{.*}} {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>}
llvm.mlir.global external @omp_decl_tar_data_host_enter() {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: global external @omp_decl_tar_data_nohost_to
// CHECK-SAME: {{.*}} {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>}
llvm.mlir.global external @omp_decl_tar_data_nohost_to() {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: global external @omp_decl_tar_data_nohost_link
// CHECK-SAME: {{.*}} {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (link)>}
llvm.mlir.global external @omp_decl_tar_data_nohost_link() {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (link)>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: global external @omp_decl_tar_data_nohost_enter
// CHECK-SAME: {{.*}} {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>}
llvm.mlir.global external @omp_decl_tar_data_nohost_enter() {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: global external @omp_decl_tar_data_any_to
// CHECK-SAME: {{.*}} {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>}
llvm.mlir.global external @omp_decl_tar_data_any_to() {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: global external @omp_decl_tar_data_any_link
// CHECK-SAME: {{.*}} {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>}
llvm.mlir.global external @omp_decl_tar_data_any_link() {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: global external @omp_decl_tar_data_any_enter
// CHECK-SAME: {{.*}} {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>}
llvm.mlir.global external @omp_decl_tar_data_any_enter() {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}
