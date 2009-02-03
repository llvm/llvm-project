;This isn't really an assembly file, its just here to run the test.
;This test just makes sure that llvm-ar can generate a table of contents for
;xpg4 style archives
;RUN: llvm-ar t %p/xpg4.a > %t1
;RUN: diff %t1 %p/xpg4.toc
