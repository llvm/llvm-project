; RUN: opt -passes=globalopt -S < %s | FileCheck %s

$comdat_global = comdat any

@comdat_global = weak_odr global i8 0, comdat($comdat_global)
@simple_global = internal global i8 0
; CHECK: @comdat_global = weak_odr global i8 0, comdat{{$}}
; CHECK: @simple_global = internal global i8 42

@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [
    { i32, void ()*, i8* } { i32 65535, void ()* @init_comdat_global, i8* @comdat_global },
    { i32, void ()*, i8* } { i32 65535, void ()* @init_simple_global, i8* null }
]
; CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }]
; CHECK: [{ i32, void ()*, i8* } { i32 65535, void ()* @init_comdat_global, i8* @comdat_global }]

define void @init_comdat_global() {
  store i8 42, i8* @comdat_global
  ret void
}
; CHECK: define void @init_comdat_global()

define internal void @init_simple_global() comdat($comdat_global) {
  store i8 42, i8* @simple_global
  ret void
}
; CHECK-NOT: @init_simple_global()

define i8* @use_simple() {
  ret i8* @simple_global
}
; CHECK: define i8* @use_simple()

define i8* @use_comdat() {
  ret i8* @comdat_global
}
; CHECK: define i8* @use_comdat()
