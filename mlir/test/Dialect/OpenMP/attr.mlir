// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: module attributes {omp.flags = #omp.flags<>} {
module attributes {omp.flags = #omp.flags<debug_kind = 0, assume_teams_oversubscription = false, assume_threads_oversubscription = false, assume_no_thread_state = false, assume_no_nested_parallelism = false>} {}

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
