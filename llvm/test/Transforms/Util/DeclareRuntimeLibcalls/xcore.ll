; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=xcore < %s | FileCheck %s

; CHECK: declare noundef i32 @fiprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) [[PRINTF_ATTRS:#[0-9]+]]
; CHECK: declare noundef i32 @iprintf(ptr noundef readonly captures(none), ...) [[PRINTF_ATTRS]]
; CHECK: declare noundef i32 @siprintf(ptr noalias noundef writeonly captures(none), ptr noundef readonly captures(none), ...) [[PRINTF_ATTRS]]

; CHECK: attributes [[PRINTF_ATTRS]] = { nofree nounwind }
