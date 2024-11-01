!baz = i64
// <- type
//     ^ type.builtin
!qux = !llvm.struct<(!baz)>
// <- type
//     ^ type
!rec = !llvm.struct<"a", (ptr<struct<"a">>)>
// <- type
//     ^ type
llvm.func @aliases() {
// <- function.builtin
//        ^ function
  "some.op"() : () -> !llvm.struct<(i32, f32, !qux)>
//                    ^ type
  "some.op"() : () -> !rec
//                    ^ type
  llvm.return
// ^ function.builtin
}
