; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes=inferattrs -S | FileCheck %s
%struct.statvfs64 = type { i32 }

; Function Attrs: norecurse uwtable
define i32 @foo() {
entry:
  %st = alloca %struct.statvfs64, align 4
  %0 = bitcast %struct.statvfs64* %st to i8*
  ret i32 0
}

; CHECK: declare i32 @statvfs64(%struct.statvfs64*){{$}}
declare i32 @statvfs64(%struct.statvfs64*)
