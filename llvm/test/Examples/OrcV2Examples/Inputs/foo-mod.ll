define i32 @foo() {
  ret i32 0
}

^0 = module: (path: "foo-mod.o", hash: (3133549885, 2087596051, 4175159200, 756405190, 968713858))
^1 = gv: (name: "foo", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 1, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0)))) ; guid = 6699318081062747564
^2 = blockcount: 0
