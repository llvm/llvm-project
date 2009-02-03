; RUN: llvm-as < %s | opt -constmerge | llvm-dis | %prcontext foo 2 | grep bar

@foo = constant i32 6           ; <i32*> [#uses=0]
@bar = constant i32 6           ; <i32*> [#uses=0]

