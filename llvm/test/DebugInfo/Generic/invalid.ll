; RUN: llvm-as -disable-output %s 2>&1 | FileCheck %s

; Make sure we emit this diagnostic only once (which means we don't visit the
; same DISubprogram twice.
; CHECK: subprogram definitions must have a compile unit
; CHECK-NEXT: !3 = distinct !DISubprogram(name: "patatino", scope: null, isLocal: false, isDefinition: true, isOptimized: false)
; CHECK-NOT: subprogram definitions must have a compile unit
; CHECK-NOT: !3 = distinct !DISubprogram(name: "patatino", scope: null, isLocal: false, isDefinition: true, isOptimized: false)
; CHECK: warning: ignoring invalid debug info

define void @tinkywinky() !dbg !3 { ret void }

!llvm.module.flags = !{!4}
!llvm.dbg.cu = !{!0}
!0 = distinct !DICompileUnit(language: 12, file: !1)
!1 = !DIFile(filename: "/home/davide", directory: "/home/davide")
!3 = distinct !DISubprogram(name: "patatino", isDefinition: true)
!4 = !{i32 2, !"Debug Info Version", i32 3}
