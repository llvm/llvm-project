; RUN: llvm-link --initial-module=%s %p/Inputs/struct-mapping.ll -S -o - | FileCheck %s

; Here we check that new type mapping algorithm correctly mapped type of internal
; member of struct.Baz to struct.Foo. Without it we'd map that type to struct.Bar, because
; it is recursively isomorphic to struct.Foo and is defined first in source file.
; CHECK: %struct.Baz = type { i64, i64, %struct.Foo }

%struct.Bar = type { i64, i64 }
%struct.Foo = type { i64, i64 }

@bar = global %struct.Bar zeroinitializer
@foo = global %struct.Foo zeroinitializer
