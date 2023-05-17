; RUN: llc < %s -mtriple=i686--
@a = extern_weak global i32             ; <ptr> [#uses=1]
@b = global ptr @a             ; <ptr> [#uses=0]

