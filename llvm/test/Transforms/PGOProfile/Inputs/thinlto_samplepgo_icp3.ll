target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fptr = external local_unnamed_addr global void ()*, align 8

; Function Attrs: norecurse nounwind uwtable
define void @_Z6updatei(i32 %i) local_unnamed_addr #0 {
entry:
  store void ()* @_ZL3foov, void ()** @fptr, align 8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define internal void @_ZL3foov() !prof !34 {
entry:
  %0 = load void ()*, void ()** @fptr, align 8
  tail call void %0(), !prof !40
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!llvm.ident = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 297016)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "b.cc", directory: "/ssd/llvm/abc/small")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{!"clang version 5.0.0 (trunk 297016)"}
!34 = !{!"function_entry_count", i64 1}
!40 = !{!"VP", i32 0, i64 3000, i64 -8789629626369651636, i64 3000}
