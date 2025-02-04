; Check that the git revision is contained in the assembly/object files

; REQUIRES: vc-rev-enabled 

; RUN: llc < %s | FileCheck %s -DREVISION=git-revision
; RUN: llc -filetype=obj < %s | FileCheck %s -DREVISION=git-revision

; CHECK: ([[REVISION]])

source_filename = "git_revision.cpp"
target datalayout = "E-m:a-Fi64-i64:64-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64-ibm-aix7.2.0.0"
