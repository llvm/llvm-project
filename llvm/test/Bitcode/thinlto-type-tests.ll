; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-dis -o - %t.o | FileCheck %s --check-prefix=DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t.o | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS
; RUN: llvm-lto -thinlto -o %t2 %t.o
; RUN: llvm-bcanalyzer -dump %t2.thinlto.bc | FileCheck --check-prefix=COMBINED %s
; RUN: llvm-dis -o - %t2.thinlto.bc | FileCheck %s --check-prefix=COMBINED-DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t2.thinlto.bc | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=COMBINED-DIS

; COMBINED: <TYPE_TESTS op0=-2012135647395072713/>
; COMBINED: <TYPE_TESTS op0=6699318081062747564 op1=-2012135647395072713/>
; COMBINED: <TYPE_TESTS op0=6699318081062747564/>

; CHECK: <TYPE_TESTS op0=6699318081062747564/>
define i1 @f() {
  %p = call i1 @llvm.type.test(i8* null, metadata !"foo")
  ret i1 %p
}

; CHECK: <TYPE_TESTS op0=6699318081062747564 op1=-2012135647395072713/>
define i1 @g() {
  %p = call i1 @llvm.type.test(i8* null, metadata !"foo")
  %q = call i1 @llvm.type.test(i8* null, metadata !"bar")
  %pq = and i1 %p, %q
  ret i1 %pq
}

; CHECK: <TYPE_TESTS op0=-2012135647395072713/>
define i1 @h() {
  %p = call i1 @llvm.type.test(i8* null, metadata !"bar")
  ret i1 %p
}

declare i1 @llvm.type.test(i8*, metadata) nounwind readnone

; DIS: ^0 = module: (path: "{{.*}}", hash: (0, 0, 0, 0, 0))
; DIS: ^1 = gv: (name: "llvm.type.test") ; guid = 608142985856744218
; DIS: ^2 = gv: (name: "h", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2, typeIdInfo: (typeTests: (16434608426314478903))))) ; guid = 8124147457056772133
; DIS: ^3 = gv: (name: "g", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 4, typeIdInfo: (typeTests: (6699318081062747564, 16434608426314478903))))) ; guid = 13146401226427987378
; DIS: ^4 = gv: (name: "f", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2, typeIdInfo: (typeTests: (6699318081062747564))))) ; guid = 14740650423002898831

; COMBINED-DIS: ^0 = module: (path: "{{.*}}thinlto-type-tests.ll.tmp.o", hash: (0, 0, 0, 0, 0))
; COMBINED-DIS: ^1 = gv: (guid: 8124147457056772133, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2, typeIdInfo: (typeTests: (16434608426314478903)))))
; COMBINED-DIS: ^2 = gv: (guid: 13146401226427987378, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 4, typeIdInfo: (typeTests: (6699318081062747564, 16434608426314478903)))))
; COMBINED-DIS: ^3 = gv: (guid: 14740650423002898831, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2, typeIdInfo: (typeTests: (6699318081062747564)))))
