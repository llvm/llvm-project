; RUN: opt -S -unique-internal-linkage-names < %s | FileCheck %s

source_filename = "foo.c"

@glob = internal global i32 0

define internal i32 @foo() {
entry:
  ret i32 0
}

; CHECK: @glob.6ae72bb15a7d1834b42ae042a58f7a4d = internal global
; CHECK: define internal i32 @foo.6ae72bb15a7d1834b42ae042a58f7a4d()
