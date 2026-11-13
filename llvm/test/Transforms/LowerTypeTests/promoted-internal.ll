; RUN: opt -passes=lowertypetests -lowertypetests-summary-action=import \
; RUN:  -lowertypetests-read-summary=%s %s -S -o - | FileCheck %s

; CHECK: @f.llvm.1234 = alias void (), ptr @f.5678.cfi
; CHECK: define hidden void @f.5678.cfi()
; CHECK: declare void @f.5678()

source_filename = "promoted-internal.ll"

; f.5678 GUID: 4670599147315008938
@f.5678 = hidden alias void (), ptr @f.llvm.1234

define void @f.llvm.1234() !type !0 !guid !{i64 1234} {
  ret void
}

!cfi.functions = !{!1}
!aliases = !{!2}
!0 = !{i64 0, !"_ZTSFvE"}
!1 = !{!"f.5678", i8 0, i64 4670599147315008938, !1}
!2 = !{!"f", !"f.5678"}

^0 = cfiFunctionDefs: ((guid: 4670599147315008938, name: "f.5678"))
