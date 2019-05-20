; Test summary parsing (tests all types/fields in various combinations).
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

; ModuleID = 'thinlto-summary.thinlto.bc'

^0 = module: (path: "thinlto-summary1.o", hash: (1369602428, 2747878711, 259090915, 2507395659, 1141468049))
^1 = module: (path: "thinlto-summary2.o", hash: (2998369023, 4283347029, 1195487472, 2757298015, 1852134156))

; Check a function that makes several calls with various profile hotness, and a
; reference (also tests forward references to function and variables in calls
; and refs).
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 10, calls: ((callee: ^15, hotness: hot), (callee: ^17, hotness: cold), (callee: ^16, hotness: none)), refs: (readonly ^13, ^14))))

; Function with a call that has relative block frequency instead of profile
; hotness.
^3 = gv: (guid: 2, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 10, calls: ((callee: ^15, relbf: 256)))))

; Summaries with different linkage types.
^4 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: internal, notEligibleToImport: 0, live: 0, dsoLocal: 1), insts: 1)))
; Make this one an alias with a forward reference to aliasee.
^5 = gv: (guid: 4, summaries: (alias: (module: ^0, flags: (linkage: private, notEligibleToImport: 0, live: 0, dsoLocal: 1), aliasee: ^14)))
^6 = gv: (guid: 5, summaries: (function: (module: ^0, flags: (linkage: available_externally, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
^7 = gv: (guid: 6, summaries: (function: (module: ^0, flags: (linkage: linkonce, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
^8 = gv: (guid: 7, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
^9 = gv: (guid: 8, summaries: (function: (module: ^0, flags: (linkage: weak_odr, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
^10 = gv: (guid: 9, summaries: (function: (module: ^0, flags: (linkage: weak, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
^11 = gv: (guid: 10, summaries: (variable: (module: ^0, flags: (linkage: common, notEligibleToImport: 0, live: 0, dsoLocal: 0), varFlags: (readonly: 0))))
; Test appending globel variable with reference (tests backward reference on
; refs).
^12 = gv: (guid: 11, summaries: (variable: (module: ^0, flags: (linkage: appending, notEligibleToImport: 0, live: 0, dsoLocal: 0), varFlags: (readonly: 0), refs: (^4))))

; Test a referenced global variable.
^13 = gv: (guid: 12, summaries: (variable: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), varFlags: (readonly: 1))))

; Test a dsoLocal variable.
^14 = gv: (guid: 13, summaries: (variable: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 1), varFlags: (readonly: 0))))

; Functions with various flag combinations (notEligibleToImport, Live,
; combinations of optional function flags).
^15 = gv: (guid: 14, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 1, live: 1, dsoLocal: 0), insts: 1)))
^16 = gv: (guid: 15, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, funcFlags: (readNone: 1, noRecurse: 1))))
; This one also tests backwards reference in calls.
^17 = gv: (guid: 16, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, funcFlags: (readOnly: 1, returnDoesNotAlias: 1), calls: ((callee: ^15)))))

; Alias summary with backwards reference to aliasee.
^18 = gv: (guid: 17, summaries: (alias: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 1), aliasee: ^14)))

; Test all types of TypeIdInfo on function summaries.
^19 = gv: (guid: 18, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 4, typeIdInfo: (typeTests: (^24, ^26)))))
^20 = gv: (guid: 19, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 8, typeIdInfo: (typeTestAssumeVCalls: (vFuncId: (^27, offset: 16))))))
^21 = gv: (guid: 20, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 5, typeIdInfo: (typeCheckedLoadVCalls: (vFuncId: (^25, offset: 16))))))
^22 = gv: (guid: 21, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 15, typeIdInfo: (typeTestAssumeConstVCalls: ((vFuncId: (^27, offset: 16), args: (42)), (vFuncId: (^27, offset: 24)))))))
^23 = gv: (guid: 22, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 5, typeIdInfo: (typeCheckedLoadConstVCalls: ((vFuncId: (^28, offset: 16), args: (42)))))))

; Test TypeId summaries:

^24 = typeid: (name: "_ZTS1C", summary: (typeTestRes: (kind: single, sizeM1BitWidth: 0)))
; Test TypeId with other optional fields (alignLog2/sizeM1/bitMask/inlineBits)
^25 = typeid: (name: "_ZTS1B", summary: (typeTestRes: (kind: inline, sizeM1BitWidth: 0, alignLog2: 1, sizeM1: 2, bitMask: 3, inlineBits: 4)))
; Test the AllOnes resolution, and all kinds of WholeProgramDevirtResolution
; types, including all optional resolution by argument kinds.
^26 = typeid: (name: "_ZTS1A", summary: (typeTestRes: (kind: allOnes, sizeM1BitWidth: 7), wpdResolutions: ((offset: 0, wpdRes: (kind: branchFunnel)), (offset: 8, wpdRes: (kind: singleImpl, singleImplName: "_ZN1A1nEi")), (offset: 16, wpdRes: (kind: indir, resByArg: (args: (1, 2), byArg: (kind: indir, byte: 2, bit: 3), args: (3), byArg: (kind: uniformRetVal, info: 1), args: (4), byArg: (kind: uniqueRetVal, info: 1), args: (5), byArg: (kind: virtualConstProp)))))))
; Test the other kinds of type test resoultions
^27 = typeid: (name: "_ZTS1D", summary: (typeTestRes: (kind: byteArray, sizeM1BitWidth: 0)))
^28 = typeid: (name: "_ZTS1E", summary: (typeTestRes: (kind: unsat, sizeM1BitWidth: 0)))

; Make sure we get back from llvm-dis essentially what we put in via llvm-as.
; CHECK: ^0 = module: (path: "thinlto-summary1.o", hash: (1369602428, 2747878711, 259090915, 2507395659, 1141468049))
; CHECK: ^1 = module: (path: "thinlto-summary2.o", hash: (2998369023, 4283347029, 1195487472, 2757298015, 1852134156))
; CHECK: ^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 10, calls: ((callee: ^15, hotness: hot), (callee: ^17, hotness: cold), (callee: ^16, hotness: none)), refs: (^14, readonly ^13))))
; CHECK: ^3 = gv: (guid: 2, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 10, calls: ((callee: ^15)))))
; CHECK: ^4 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: internal, notEligibleToImport: 0, live: 0, dsoLocal: 1), insts: 1)))
; CHECK: ^5 = gv: (guid: 4, summaries: (alias: (module: ^0, flags: (linkage: private, notEligibleToImport: 0, live: 0, dsoLocal: 1), aliasee: ^14)))
; CHECK: ^6 = gv: (guid: 5, summaries: (function: (module: ^0, flags: (linkage: available_externally, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; CHECK: ^7 = gv: (guid: 6, summaries: (function: (module: ^0, flags: (linkage: linkonce, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; CHECK: ^8 = gv: (guid: 7, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; CHECK: ^9 = gv: (guid: 8, summaries: (function: (module: ^0, flags: (linkage: weak_odr, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; CHECK: ^10 = gv: (guid: 9, summaries: (function: (module: ^0, flags: (linkage: weak, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))
; CHECK: ^11 = gv: (guid: 10, summaries: (variable: (module: ^0, flags: (linkage: common, notEligibleToImport: 0, live: 0, dsoLocal: 0), varFlags: (readonly: 0))))
; CHECK: ^12 = gv: (guid: 11, summaries: (variable: (module: ^0, flags: (linkage: appending, notEligibleToImport: 0, live: 0, dsoLocal: 0), varFlags: (readonly: 0), refs: (^4))))
; CHECK: ^13 = gv: (guid: 12, summaries: (variable: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), varFlags: (readonly: 1))))
; CHECK: ^14 = gv: (guid: 13, summaries: (variable: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 1), varFlags: (readonly: 0))))
; CHECK: ^15 = gv: (guid: 14, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 1, live: 1, dsoLocal: 0), insts: 1)))
; CHECK: ^16 = gv: (guid: 15, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, funcFlags: (readNone: 1, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 0))))
; CHECK: ^17 = gv: (guid: 16, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, funcFlags: (readNone: 0, readOnly: 1, noRecurse: 0, returnDoesNotAlias: 1, noInline: 0), calls: ((callee: ^15)))))
; CHECK: ^18 = gv: (guid: 17, summaries: (alias: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 1), aliasee: ^14)))
; CHECK: ^19 = gv: (guid: 18, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 4, typeIdInfo: (typeTests: (^24, ^26)))))
; CHECK: ^20 = gv: (guid: 19, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 8, typeIdInfo: (typeTestAssumeVCalls: (vFuncId: (^27, offset: 16))))))
; CHECK: ^21 = gv: (guid: 20, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 5, typeIdInfo: (typeCheckedLoadVCalls: (vFuncId: (^25, offset: 16))))))
; CHECK: ^22 = gv: (guid: 21, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 15, typeIdInfo: (typeTestAssumeConstVCalls: ((vFuncId: (^27, offset: 16), args: (42)), (vFuncId: (^27, offset: 24)))))))
; CHECK: ^23 = gv: (guid: 22, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 5, typeIdInfo: (typeCheckedLoadConstVCalls: ((vFuncId: (^28, offset: 16), args: (42)))))))
; CHECK: ^24 = typeid: (name: "_ZTS1C", summary: (typeTestRes: (kind: single, sizeM1BitWidth: 0))) ; guid = 1884921850105019584
; CHECK: ^25 = typeid: (name: "_ZTS1B", summary: (typeTestRes: (kind: inline, sizeM1BitWidth: 0, alignLog2: 1, sizeM1: 2, bitMask: 3, inlineBits: 4))) ; guid = 6203814149063363976
; CHECK: ^26 = typeid: (name: "_ZTS1A", summary: (typeTestRes: (kind: allOnes, sizeM1BitWidth: 7), wpdResolutions: ((offset: 0, wpdRes: (kind: branchFunnel)), (offset: 8, wpdRes: (kind: singleImpl, singleImplName: "_ZN1A1nEi")), (offset: 16, wpdRes: (kind: indir, resByArg: (args: (1, 2), byArg: (kind: indir, byte: 2, bit: 3), args: (3), byArg: (kind: uniformRetVal, info: 1), args: (4), byArg: (kind: uniqueRetVal, info: 1), args: (5), byArg: (kind: virtualConstProp))))))) ; guid = 7004155349499253778
; CHECK: ^27 = typeid: (name: "_ZTS1D", summary: (typeTestRes: (kind: byteArray, sizeM1BitWidth: 0))) ; guid = 9614786172484273522
; CHECK: ^28 = typeid: (name: "_ZTS1E", summary: (typeTestRes: (kind: unsat, sizeM1BitWidth: 0))) ; guid = 17437243864166745132

; Make sure parsing of a non-summary entry containing a ":" does not fail
; after summary parsing, which handles colons differently.
!0 = !DIFile(filename: "thinlto-summary.ll", directory: "", source: "")
