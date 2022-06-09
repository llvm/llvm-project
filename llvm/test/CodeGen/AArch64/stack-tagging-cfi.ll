; RUN: llc < %s -mtriple=aarch64 -mattr=+mte | FileCheck %s

declare void @use32(ptr)

define void @WithUnwind() sanitize_memtag {
entry:
; CHECK-LABEL: WithUnwind:
; CHECK: .cfi_mte_tagged_frame
; CHECK: stg
  %x = alloca i32, align 4
  call void @use32(i32* %x)
  ret void
}

define void @NoUnwind() sanitize_memtag nounwind {
entry:
; CHECK-LABEL: NoUnwind:
; CHECK-NOT: .cfi_mte_tagged_frame
; CHECK: stg
  %x = alloca i32, align 4
  call void @use32(i32* %x)
  ret void
}

define void @NoUnwindUwTable() sanitize_memtag nounwind uwtable {
entry:
; CHECK-LABEL: NoUnwindUwTable:
; CHECK: .cfi_mte_tagged_frame
; CHECK: stg
  %x = alloca i32, align 4
  call void @use32(i32* %x)
  ret void
}

define void @NoMemtag() {
entry:
; CHECK-LABEL: NoMemtag:
; CHECK-NOT: .cfi_mte_tagged_frame
; CHECK-NOT: stg
  %x = alloca i32, align 4
  call void @use32(i32* %x)
  ret void
}
