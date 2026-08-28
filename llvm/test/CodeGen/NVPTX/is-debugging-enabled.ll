; RUN: llc -mtriple=nvptx64-nvidia-cuda < %s | FileCheck %s

define i1 @query() {
; CHECK-LABEL: query(
; CHECK:       st.param.b32 [func_retval0], 0;
; CHECK-NEXT:  ret;
  %enabled = call i1 @llvm.is.debugging.enabled()
  ret i1 %enabled
}

declare i1 @llvm.is.debugging.enabled()
