target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@_g = dso_local global i32 0, align 4
@llvm.compiler.used = appending global [1 x ptr] [ptr @_g], section "llvm.metadata"

define dso_local i32 @foo(i32 noundef %0) #0 {
  %2 = add nsw i32 %0, 42
  store i32 %2, ptr @_g, align 4
  ret i32 %2
}

attributes #0 = { noinline }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"ThinLTO", i32 0}
!1 = !{i32 1, !"EnableSplitLTOUnit", i32 1}

^0 = module: (path: "/tmp/func2.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "foo", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 3, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), refs: (^3)))) ; guid = 6699318081062747564
^2 = gv: (name: "llvm.compiler.used", summaries: (variable: (module: ^0, flags: (linkage: appending, visibility: default, notEligibleToImport: 1, live: 1, dsoLocal: 0, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 0), refs: (^3)))) ; guid = 9610627770985738006
^3 = gv: (name: "_g", summaries: (variable: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0), varFlags: (readonly: 1, writeonly: 1, constant: 0)))) ; guid = 9713702464056781075
^4 = flags: 8
^5 = blockcount: 0
