; ModuleID = '/tmp/devirt-index.bc'
source_filename = "/tmp/devirt-index.bc"

^0 = module: (path: "/tmp/main.bc", hash: (3499594384, 1671013073, 3271036935, 1830411232, 59290952))
^1 = module: (path: "/tmp/foo.bc", hash: (1981453201, 1990260332, 4054522231, 886164300, 2116061388))
^2 = module: (path: "/tmp/bar.bc", hash: (1315792037, 3870713320, 284974409, 169291533, 3565750560))
^3 = module: (path: "[Regular LTO]", hash: (0, 0, 0, 0, 0))
^4 = gv: (guid: 7004155349499253778, summaries: (variable: (module: ^2, flags: (linkage: linkonce_odr, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 1))))
^5 = gv: (guid: 7112837063505133550, summaries: (variable: (module: ^2, flags: (linkage: linkonce_odr, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 1), refs: (^4))))
^6 = gv: (guid: 12105754951942688208, summaries: (variable: (module: ^3, flags: (linkage: linkonce_odr, notEligibleToImport: 1, live: 1, dsoLocal: 1, canAutoHide: 1), varFlags: (readonly: 0, writeonly: 0, constant: 1), refs: (^5, ^7))))
^7 = gv: (guid: 13351721993301222997, summaries: (function: (module: ^2, flags: (linkage: linkonce_odr, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 1), insts: 1), function: (module: ^3, flags: (linkage: available_externally, notEligibleToImport: 1, live: 1, dsoLocal: 1, canAutoHide: 0), insts: 1)))
^8 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0), insts: 2, calls: ((callee: ^10)))))
^9 = gv: (guid: 16692224328168775211, summaries: (function: (module: ^1, flags: (linkage: external, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0), insts: 9, typeIdInfo: (typeTestAssumeConstVCalls: ((vFuncId: (guid: 7004155349499253778, offset: 0)))))))
^10 = gv: (guid: 17377440600225628772, summaries: (function: (module: ^2, flags: (linkage: external, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0), insts: 9, calls: ((callee: ^9)), refs: (^6))))
