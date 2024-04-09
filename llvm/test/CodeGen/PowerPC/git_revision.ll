; Check that the git revision is contained in the assembly/object files

; RUN: cd %S && git rev-parse HEAD > %t

; RUN: llc < %s > %t.a.o
; RUN: sed -n -e "/$(cat %t)/p" %t.a.o

; RUN: llc -filetype=obj < %s > %t.b.o
; RUN: sed -n -e "/$(cat %t)/p" %t.b.o

source_filename = "git_revision.cpp"
target datalayout = "E-m:a-Fi64-i64:64-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64-ibm-aix7.2.0.0"
