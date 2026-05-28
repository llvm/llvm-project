; RUN: not llc < %s -mtriple=arm64-apple-darwin 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=arm64-linux-gnueabi 2>&1 | FileCheck %s

define i32 @get_stack() nounwind {
entry:
; CHECK: error: <unknown>:0:0: invalid register "notareg" for llvm.read_register
  %sp = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %sp
}

define i64 @read_invalid() nounwind {
; CHECK: error: <unknown>:0:0: invalid register "1:2:3:4:5" for llvm.read_register
entry:
  %reg = call i64 @llvm.read_register.i64(metadata !1)
  ret i64 %reg
}

define void @write_invalid(i64 %x) nounwind {
; CHECK: error: <unknown>:0:0: invalid register "1:2:3:4:5" for llvm.write_register
entry:
  call void @llvm.write_register.i64(metadata !1, i64 %x)
  ret void
}

!0 = !{!"notareg\00"}
!1 = !{!"1:2:3:4:5"}