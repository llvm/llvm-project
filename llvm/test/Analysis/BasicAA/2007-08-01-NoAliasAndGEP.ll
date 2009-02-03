; RUN: llvm-as %s -o - | opt -basicaa -aa-eval -print-all-alias-modref-info -disable-output |& grep {9 no alias}
; RUN: llvm-as %s -o - | opt -basicaa -aa-eval -print-all-alias-modref-info -disable-output |& grep {6 may alias}
; RUN: llvm-as %s -o - | opt -basicaa -aa-eval -print-all-alias-modref-info -disable-output |& grep {MayAlias:.*i32\\* %Ipointer, i32\\* %Jpointer}

define void @foo(i32* noalias %p, i32* noalias %q, i32 %i, i32 %j) {
  %Ipointer = getelementptr i32* %p, i32 %i
  %qi = getelementptr i32* %q, i32 %i
  %Jpointer = getelementptr i32* %p, i32 %j
  %qj = getelementptr i32* %q, i32 %j
  store i32 0, i32* %p
  store i32 0, i32* %Ipointer
  store i32 0, i32* %Jpointer
  store i32 0, i32* %q
  store i32 0, i32* %qi
  store i32 0, i32* %qj
  ret void
}
