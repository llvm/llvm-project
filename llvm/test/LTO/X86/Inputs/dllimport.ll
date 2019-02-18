; ModuleID = 'b.obj'
source_filename = "b.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

; Function Attrs: norecurse nounwind readnone sspstrong uwtable
define dso_local i32 @"?foo@@YAHXZ"() local_unnamed_addr {
entry:
  ret i32 42
}

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ThinLTO", i32 0}

^0 = module: (path: "b.obj", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "?foo@@YAHXZ", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 1, live: 0, dsoLocal: 1), insts: 1, funcFlags: (readNone: 1, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0)))) ; guid = 2709792123250749187
