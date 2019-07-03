; RUN: llc %s -o - | FileCheck %s
target triple="x86_64-apple-ios13.0-macabi"
; CHECK: .build_version maccatalyst, 13, 0
