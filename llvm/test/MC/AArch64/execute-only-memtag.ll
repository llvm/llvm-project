; RUN: llc %s -mtriple=aarch64-linux-android31 -filetype=obj -o %t.o
; RUN: llvm-readelf -r %t.o | FileCheck %s

; CHECK:      Relocation section '.rela.memtag.globals.static' at offset {{.*}} contains 1 entries:
; CHECK-NEXT:      Type      {{.*}} Symbol's Name
; CHECK-NEXT: R_AARCH64_NONE {{.*}} global

@global = global i32 1, sanitize_memtag

define void @foo() {
  ret void
}

define void @bar() #0 {
  ret void
}

attributes #0 = { "target-features"="+execute-only" }
