;; Check that if the address of a weak function is only taken through an alias,
;; it is still added to a list of exported functions and @llvm.type.test() is
;; lowered to an actual check against the generated CFI jumptable.

RUN: rm -rf %t.dir && split-file %s %t.dir && cd %t.dir
RUN: opt test.ll --thinlto-bc --thinlto-split-lto-unit -o test.bc
RUN: llvm-modextract test.bc -n 0 -o test0.bc
RUN: llvm-modextract test.bc -n 1 -o test1.bc

;; Check that a CFI jumptable is generated.
RUN: opt test1.bc -passes=lowertypetests -lowertypetests-read-summary=in.ll \
RUN:   -lowertypetests-summary-action=export -lowertypetests-write-summary=exported.yaml \
RUN:   -S -o - | FileCheck %s --check-prefix=REGULAR
REGULAR: @__typeid__ZTSFvvE_global_addr = hidden alias i8, ptr @.cfi.jumptable
REGULAR: @f = alias [8 x i8], ptr @.cfi.jumptable
REGULAR: define private void @.cfi.jumptable()

;; CHECK that @llvm.type.test() is lowered to an actual check.
RUN: opt test0.bc -passes=lowertypetests -lowertypetests-read-summary=exported.yaml \
RUN:   -lowertypetests-summary-action=import -S -o - | FileCheck %s --check-prefix=THIN
THIN:      define i1 @test() !guid !{{.*}} {
THIN-NEXT:   %1 = icmp eq i64 ptrtoint (ptr @alias to i64), ptrtoint (ptr @__typeid__ZTSFvvE_global_addr to i64)
THIN-NEXT:   ret i1 %1
THIN-NEXT: }

;--- test.ll
target triple = "x86_64-pc-linux-gnu"

@alias = alias void(), ptr @f

define weak void @f() !type !0 {
  ret void
}

define i1 @test() {
  %1 = call i1 @llvm.type.test(ptr nonnull @alias, metadata !"_ZTSFvvE")
  ret i1 %1
}

declare i1 @llvm.type.test(ptr, metadata)

!0 = !{i64 0, !"_ZTSFvvE"}
;--- in.ll
^0 = module: (path: "test.bc", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 8346051122425466633,  summaries: (function: (module: ^0, flags: (live: 1), insts: 1, refs: (^2), typeIdInfo: (typeTests: (9080559750644022485)))))
^2 = gv: (guid: 5833419078793185394,  summaries: (alias:    (module: ^0, flags: (live: 1), aliasee: ^3)))
^3 = gv: (guid: 14740650423002898831, summaries: (function: (module: ^0, flags: (linkage: weak, live: 1), insts: 1)))
