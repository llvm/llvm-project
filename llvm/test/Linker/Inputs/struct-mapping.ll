%struct.Baz = type { i64, i64, %struct.Foo }
%struct.Foo = type { i64, i64 }

@baz = global %struct.Baz zeroinitializer
