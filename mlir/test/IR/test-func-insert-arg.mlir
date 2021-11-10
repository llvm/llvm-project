// RUN: mlir-opt %s -test-func-insert-arg -split-input-file | FileCheck %s

// CHECK: func @f(%arg0: i1 {test.A})
func @f() attributes {test.insert_args = [
  [0, i1, {test.A}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: i1 {test.A}, %arg1: i2 {test.B})
func @f(%arg0: i2 {test.B}) attributes {test.insert_args = [
  [0, i1, {test.A}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: i1 {test.A}, %arg1: i2 {test.B})
func @f(%arg0: i1 {test.A}) attributes {test.insert_args = [
  [1, i2, {test.B}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: i1 {test.A}, %arg1: i2 {test.B}, %arg2: i3 {test.C})
func @f(%arg0: i1 {test.A}, %arg1: i3 {test.C}) attributes {test.insert_args = [
  [1, i2, {test.B}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: i1 {test.A}, %arg1: i2 {test.B}, %arg2: i3 {test.C})
func @f(%arg0: i2 {test.B}) attributes {test.insert_args = [
  [0, i1, {test.A}],
  [1, i3, {test.C}]]} {
  return
}

// -----

// CHECK: func @f(%arg0: i1 {test.A}, %arg1: i2 {test.B}, %arg2: i3 {test.C})
func @f(%arg0: i3 {test.C}) attributes {test.insert_args = [
  [0, i1, {test.A}],
  [0, i2, {test.B}]]} {
  return
}
