; RUN: opt < %s -internalize -internalize-public-api-list 'bar?,_*,*_,[ab]' -S | FileCheck %s

; CHECK: @foo = internal global
@foo = global i32 0

; CHECK: @bar_ = global
@bar_ = global i32 0

; CHECK: @_foo = global
@_foo = global i32 0

; CHECK: @foo_ = global
@foo_ = global i32 0

; CHECK: @a = global
@a = global i32 0

; CHECK: @b = global
@b = global i32 0

; CHECK: @c = internal global
@c = global i32 0
