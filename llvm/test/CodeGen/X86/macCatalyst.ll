; RUN: llc %s -o - | FileCheck %s
target triple="x86_64-apple-ios13.0-macabi"
; CHECK: .build_version macCatalyst, 13, 0
