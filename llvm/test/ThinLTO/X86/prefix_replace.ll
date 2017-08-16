; Check that changing the output path via prefix-replace works
; Use of '/' in paths created here make this unsuitable for Windows.
; RUN: mkdir -p %t/oldpath
; RUN: opt -module-summary %s -o %t/oldpath/prefix_replace.o
; Ensure that there is no existing file at the new path, so we properly
; test the creation of the new file there.
; RUN: rm -f %t/newpath/prefix_replace.o.thinlto.bc

; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t/oldpath/prefix_replace.o
; RUN: llvm-lto -thinlto-action=distributedindexes -thinlto-prefix-replace="%t/oldpath/;%t/newpath/" -thinlto-index %t.index.bc %t/oldpath/prefix_replace.o

; RUN: ls %t/newpath/prefix_replace.o.thinlto.bc

define void @f() {
entry:
  ret void
}
