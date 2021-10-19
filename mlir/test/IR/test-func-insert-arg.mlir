// RUN: mlir-opt %s -test-func-insert-arg -split-input-file | FileCheck %s

// CHECK: func @f(%arg0: f32 {test.A})
func @f() attributes {test.insert_args = [
  [0, f32, {test.A}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A}, %arg1: f32 {test.B})
func @f(%arg0: f32 {test.B}) attributes {test.insert_args = [
  [0, f32, {test.A}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A}, %arg1: f32 {test.B})
func @f(%arg0: f32 {test.A}) attributes {test.insert_args = [
  [1, f32, {test.B}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A}, %arg1: f32 {test.B}, %arg2: f32 {test.C})
func @f(%arg0: f32 {test.A}, %arg1: f32 {test.C}) attributes {test.insert_args = [
  [1, f32, {test.B}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A}, %arg1: f32 {test.B}, %arg2: f32 {test.C})
func @f(%arg0: f32 {test.B}) attributes {test.insert_args = [
  [0, f32, {test.A}],
  [1, f32, {test.C}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: f32 {test.A}, %arg1: f32 {test.B}, %arg2: f32 {test.C})
func @f(%arg0: f32 {test.C}) attributes {test.insert_args = [
  [0, f32, {test.A}],
  [0, f32, {test.B}]]} {
  return
}
