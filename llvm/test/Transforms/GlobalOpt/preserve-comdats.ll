; RUN: opt -passes=globalopt -S < %s | FileCheck %s

$comdat_global = comdat any

@comdat_global = weak_odr global i8 0, comdat($comdat_global)
@simple_global = internal global i8 0
; CHECK: @comdat_global = weak_odr global i8 0, comdat{{$}}
; CHECK: @simple_global = internal global i8 42

@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [
    { i32, ptr, ptr } { i32 65535, ptr @init_comdat_global, ptr @comdat_global },
    { i32, ptr, ptr } { i32 65535, ptr @init_simple_global, ptr null }
]
; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
; CHECK: [{ i32, ptr, ptr } { i32 65535, ptr @init_comdat_global, ptr @comdat_global }]

define void @init_comdat_global() {
  store i8 42, ptr @comdat_global
  ret void
}
; CHECK: define void @init_comdat_global()

define internal void @init_simple_global() comdat($comdat_global) {
  store i8 42, ptr @simple_global
  ret void
}
; CHECK-NOT: @init_simple_global()

define ptr @use_simple() {
  ret ptr @simple_global
}
; CHECK: define ptr @use_simple()

define ptr @use_comdat() {
  ret ptr @comdat_global
}
; CHECK: define ptr @use_comdat()
