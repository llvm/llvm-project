; Even if value ids come out of order summary assembly should be parsed correctly
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

; CHECK-DAG: ^[[VTBL:[0-9]+]] = gv: {{.*}} "_ZTVN3FooE", {{.*}}virtFunc: ^[[VFN:[0-9]+]]
; CHECK-DAG: ^{{[0-9]+}} = typeidCompatibleVTable: {{.*}}name: "_ZTSN3FooE",{{.*}}(offset: 16, ^[[VTBL]])
; CHECK-DAG: ^{{[0-9]+}} = gv: {{.*}}name: "_ZTSN3FooE"
; CHECK-DAG: ^[[VFN]] = gv: {{.*}} "_Z3barv"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZTSN3FooE = comdat any

@_ZTSN3FooE = linkonce_odr constant [7 x i8] c"N3FooE\00", comdat, align 1
@"_ZTVN3FooE" = internal unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr @"_Z3barv"] }, align 8

define internal i32 @"_Z3barv"() {
  ret i32 0
}

^0 = module: (path: "index-value-order.ll", hash: (0, 0, 0, 0, 0))
^9 = gv: (name: "_ZTVN3FooE", summaries: (variable: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), varFlags: (readonly: 1, writeonly: 0, constant: 1, vcall_visibility: 0), vTableFuncs: ((virtFunc: ^3, offset: 16)))))
^4 = gv: (name: "_ZTSN3FooE", summaries: (variable: (module: ^0, flags: (linkage: linkonce_odr, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 1))))
^3 = gv: (name: "_Z3barv", summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 1, funcFlags: (readNone: 1, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0))))
^2 = typeidCompatibleVTable: (name: "_ZTSN3FooE", summary: ((offset: 16, ^9)))
