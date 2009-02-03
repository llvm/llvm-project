; RUN: llvm-as < %s | opt -anders-aa -gvn -instcombine | llvm-dis \
; RUN: | grep {ret i1 true}

@G = internal global i32* null
declare i32 *@ext()

define i1 @bar() {
  %V1 = load i32** @G
  %X2 = call i32 *@ext()
  %V2 = load i32** @G
  store i32* %X2, i32** @G

  %C = icmp eq i32* %V1, %V2
  ret i1 %C
}
