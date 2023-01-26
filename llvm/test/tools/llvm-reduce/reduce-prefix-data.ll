; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=function-data --test FileCheck --test-arg --check-prefix=CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=CHECK,RESULT %s < %t

@gv = linkonce_odr global i32 1

; CHECK: define void @drop_prefix_i32_const()
define void @drop_prefix_i32_const() prefix i32 1 {
  ret void
}

; CHECK: define void @keep_prefix_i32_const() prefix i32 1 {
define void @keep_prefix_i32_const() prefix i32 1 {
  ret void
}

; CHECK: define void @drop_prefix_ptr_global()
define void @drop_prefix_ptr_global() prefix ptr @gv {
  ret void
}

; CHECK: define void @keep_prefix_ptr_global() prefix ptr @gv {
define void @keep_prefix_ptr_global() prefix ptr @gv {
  ret void
}

; Make sure there's no invalid reduction if the prefix data is really
; accessed
; CHECK: define i32 @drop_uses_prefix_i32_const()
define i32 @drop_uses_prefix_i32_const() prefix i32 1 {
  %gep = getelementptr inbounds i32, ptr @drop_uses_prefix_i32_const, i32 -1
  %load = load i32, ptr %gep
  ret i32 %load
}

; CHECK: define ptr @drop_uses_prefix_gv(
define ptr @drop_uses_prefix_gv() prefix ptr @gv {
  %gep = getelementptr inbounds i32, ptr @drop_uses_prefix_gv, i32 -1
  %load = load ptr, ptr %gep
  ret ptr %load
}


; RESULT: declare void @declaration_prefix_i32_const(){{$}}
declare void @declaration_prefix_i32_const() prefix i32 2

; RESULT: void @declaration_prefix_ptr_gv(){{$}}
declare void @declaration_prefix_ptr_gv() prefix ptr @gv
