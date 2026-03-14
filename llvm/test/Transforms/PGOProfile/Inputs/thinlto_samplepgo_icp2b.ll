target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_ZL3barv() #1 {
entry:
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  call void @dummy()
  ret void
}

define internal void @dummy() {
entry:
    ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!llvm.ident = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 297016)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "c.cc", directory: "/ssd/llvm/abc/small")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{!"clang version 5.0.0 (trunk 297016)"}
