; Check that changing the output path via prefix-replace works
; Use of '/' in paths created here make this unsuitable for Windows.
; RUN: mkdir -p %T/oldpath
; RUN: opt -module-summary %s -o %T/oldpath/prefix_replace.o
; Ensure that there is no existing file at the new path, so we properly
; test the creation of the new file there.
; RUN: rm -f %T/newpath/prefix_replace.o.thinlto.bc

; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %T/oldpath/prefix_replace.o
; RUN: llvm-lto -thinlto-action=distributedindexes -thinlto-prefix-replace="%T/oldpath/;%T/newpath/" -thinlto-index %t.index.bc %T/oldpath/prefix_replace.o

; RUN: ls %T/newpath/prefix_replace.o.thinlto.bc

define void @f() {
entry:
  ret void
}
