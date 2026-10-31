; REQUIRES: asserts
; RUN: opt -disable-output -enable-profcheck-in-pipeline -print-pipeline-passes -passes="default<O0>" < %s 2>&1 | FileCheck %s
; RUN: opt -disable-output -enable-profcheck-in-pipeline -print-pipeline-passes -passes="default<O3>" < %s 2>&1 | FileCheck %s
; RUN: opt -disable-output -enable-profcheck-in-pipeline -print-pipeline-passes -passes="lto<O3>" < %s 2>&1 | FileCheck %s
; RUN: opt -disable-output -enable-profcheck-in-pipeline -print-pipeline-passes -passes="lto-pre-link<O2>" < %s 2>&1 | FileCheck %s
; RUN: opt -disable-output -enable-profcheck-in-pipeline -print-pipeline-passes -passes="lto-pre-link<O3>" < %s 2>&1 | FileCheck %s
; RUN: opt -disable-output -enable-profcheck-in-pipeline -print-pipeline-passes -passes="thinlto<O3>" < %s 2>&1 | FileCheck %s
; RUN: opt -disable-output -enable-profcheck-in-pipeline -print-pipeline-passes -passes="thinlto-pre-link<O2>" < %s 2>&1 | FileCheck %s

; CHECK: prof-inject
; CHECK: prof-verify
