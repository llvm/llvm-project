; RUN: llc < %s -mtriple=i686--
@a = extern_weak global i32             ; <i32*> [#uses=1]
@b = global i32* @a             ; <i32**> [#uses=0]

