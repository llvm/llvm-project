; RUN: opt -S -unique-internal-linkage-names < %s | FileCheck %s
; RUN: opt -S -passes=unique-internal-linkage-names < %s | FileCheck %s

source_filename = "foo.c"

@glob = internal global i32 0

define internal i32 @foo() {
entry:
  ret i32 0
}

; CHECK: @glob.__uniq.142098474322525230676991677820000238157 = internal global
; CHECK: define internal i32 @foo.__uniq.142098474322525230676991677820000238157()
