; RUN: opt < %s -mtriple=x86_64-apple-darwin -frame-pointer=all -S | FileCheck -check-prefixes=ALL,CHECK %s
; RUN: opt < %s -mtriple=x86_64-apple-darwin -frame-pointer=none -S | FileCheck -check-prefixes=NONE,CHECK %s
; RUN: opt < %s -mtriple=x86_64-apple-darwin -frame-pointer=non-leaf -S | FileCheck -check-prefixes=NONLEAF,CHECK %s

; Check behavior of -frame-pointer flag and frame-pointer atttribute.

; CHECK: @no_frame_pointer_attr() [[VARATTR:#[0-9]+]] {
define i32 @no_frame_pointer_attr() #0 {
entry:
  ret i32 0
}

; CHECK: @frame_pointer_attr_all() [[ALL_ATTR:#[0-9]+]] {
define i32 @frame_pointer_attr_all() #1 {
entry:
  ret i32 0
}

; CHECK: @frame_pointer_attr_none() [[NONE_ATTR:#[0-9]+]] {
define i32 @frame_pointer_attr_none() #2 {
entry:
  ret i32 0
}

; CHECK: @frame_pointer_attr_leaf() [[NONLEAF_ATTR:#[0-9]+]] {
define i32 @frame_pointer_attr_leaf() #3 {
entry:
  ret i32 0
}

; ALL-DAG: attributes [[VARATTR]] = { nounwind "frame-pointer"="all" }
; ALL-DAG: attributes [[NONE_ATTR]] = { nounwind "frame-pointer"="none" }
; ALL-DAG: attributes [[NONLEAF_ATTR]] = { nounwind "frame-pointer"="non-leaf" }
; ALL-NOT: attributes

; NONE-DAG: attributes [[VARATTR]] = { nounwind "frame-pointer"="none" }
; NONE-DAG: attributes [[ALL_ATTR]] = { nounwind "frame-pointer"="all" }
; NONE-DAG: attributes [[NONLEAF_ATTR]] = { nounwind "frame-pointer"="non-leaf" }
; NONE-NOT: attributes

; NONLEAF-DAG: attributes [[VARATTR]] = { nounwind "frame-pointer"="non-leaf" }
; NONLEAF-DAG: attributes [[ALL_ATTR]] = { nounwind "frame-pointer"="all" }
; NONLEAF-DAG: attributes [[NONE_ATTR]] = { nounwind "frame-pointer"="none" }
; NONLEAF-NOT: attributes


attributes #0 = { nounwind }
attributes #1 = { nounwind "frame-pointer"="all" }
attributes #2 = { nounwind "frame-pointer"="none" }
attributes #3 = { nounwind "frame-pointer"="non-leaf" }
