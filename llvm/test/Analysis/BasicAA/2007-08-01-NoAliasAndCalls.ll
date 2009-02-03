; RUN: llvm-as %s -o - | opt -basicaa -aa-eval -print-all-alias-modref-info -disable-output |& grep {MayAlias:.*i32\\* %., i32\\* %.} | grep {%x} | grep {%y}

declare i32* @unclear(i32* %a)

define void @foo(i32* noalias %x) {
  %y = call i32* @unclear(i32* %x)
  store i32 0, i32* %x
  store i32 0, i32* %y
  ret void
}
