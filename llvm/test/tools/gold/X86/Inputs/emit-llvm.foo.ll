target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @_Z3foov() #0 {
  ret i32 0
}

^0 = module: (path: "foo.o", hash: (3786635673, 3275786284, 1544384821, 2103351884, 2354656665))
^1 = gv: (name: "_Z3foov", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 1, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0)))) ; guid = 9191153033785521275
