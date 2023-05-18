; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=function-data --test FileCheck --test-arg --check-prefix=CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=CHECK,RESULT %s < %t

@gv = linkonce_odr global i32 1

; CHECK: define void @drop_prologue_const_array()
define void @drop_prologue_const_array() prologue [2 x i32] [i32 1, i32 2] {
  ret void
}

; CHECK: define void @keep_prologue_const_array() prologue i32 1 {
define void @keep_prologue_const_array() prologue i32 1 {
  ret void
}

; CHECK: define void @drop_prologue_ptr_global()
define void @drop_prologue_ptr_global() prologue ptr @gv {
  ret void
}

; CHECK: define void @keep_prologue_ptr_global() prologue ptr @gv {
define void @keep_prologue_ptr_global() prologue ptr @gv {
  ret void
}

; Make sure there's no invalid reduction if the prologue data is really
; accessed
; CHECK: define i32 @drop_uses_prologue_const_array()
define i32 @drop_uses_prologue_const_array() prologue [2 x i32] [i32 1, i32 2] {
  %gep = getelementptr inbounds i32, ptr @drop_uses_prologue_const_array, i32 -1
  %load = load i32, ptr %gep
  ret i32 %load
}

; CHECK: define ptr @drop_uses_prologue_gv(
define ptr @drop_uses_prologue_gv() prologue ptr @gv {
  %gep = getelementptr inbounds i32, ptr @drop_uses_prologue_gv, i32 -1
  %load = load ptr, ptr %gep
  ret ptr %load
}


; RESULT: declare void @declaration_prologue_const_array(){{$}}
declare void @declaration_prologue_const_array() prologue [2 x i32] [i32 1, i32 2]

; RESULT: void @declaration_prologue_ptr_gv(){{$}}
declare void @declaration_prologue_ptr_gv() prologue ptr @gv
