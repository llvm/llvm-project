; REQUIRES: x86-registered-target
;; Check that we infer the correct datalayout from a target triple
; RUN: opt -mtriple=i386-linux-gnu -S -passes=no-op-module < %s | FileCheck %s --check-prefix=LINUX
; RUN: opt -mtriple=i386-apple-darwin -S -passes=no-op-module < %s | FileCheck %s --check-prefix=DARWIN
; RUN: opt -mtriple=i386-windows-msvc -S -passes=no-op-module < %s | FileCheck %s --check-prefix=WINDOWS

target datalayout = ""
; LINUX: target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i128:128-f64:32:64-f80:32-n8:16:32-S128"
; LINUX: target triple = "i386-unknown-linux-gnu"
; DARWIN: target datalayout = "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-i128:128-f64:32:64-f80:128-n8:16:32-S128"
; DARWIN: target triple = "i386-apple-darwin"
; WINDOWS: target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32-a:0:32-S32"
; WINDOWS: target triple = "i386-unknown-windows-msvc"
