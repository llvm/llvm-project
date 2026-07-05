// RUN: mlir-opt %s -test-func-insert-result -split-input-file | FileCheck %s

// CHECK: func private @f() -> (f32 {test.A})
func.func private @f() attributes {test.insert_results = [
  [0, f32, {test.A}]]}

// -----

// CHECK: func private @f() -> (f32 {test.A}, f32 {test.B})
func.func private @f() -> (f32 {test.B}) attributes {test.insert_results = [
  [0, f32, {test.A}]]}

// -----

// CHECK: func private @f() -> (f32 {test.A}, f32 {test.B})
func.func private @f() -> (f32 {test.A}) attributes {test.insert_results = [
  [1, f32, {test.B}]]}

// -----

// CHECK: func private @f() -> (f32 {test.A}, f32 {test.B}, f32 {test.C})
func.func private @f() -> (f32 {test.A}, f32 {test.C}) attributes {test.insert_results = [
  [1, f32, {test.B}]]}

// -----

// CHECK: func private @f() -> (f32 {test.A}, f32 {test.B}, f32 {test.C})
func.func private @f() -> (f32 {test.B}) attributes {test.insert_results = [
  [0, f32, {test.A}],
  [1, f32, {test.C}]]}

// -----

// CHECK: func private @f() -> (f32 {test.A}, f32 {test.B}, f32 {test.C})
func.func private @f() -> (f32 {test.C}) attributes {test.insert_results = [
  [0, f32, {test.A}],
  [0, f32, {test.B}]]}
