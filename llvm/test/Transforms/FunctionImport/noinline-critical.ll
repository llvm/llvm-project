; A module can request the import of a function by listing its GUID in the
; function_entry_count metadata of one of its functions, which turns the call
; edge into a critical one. A noinline callee is still not imported, unless
; -import-critical-noinline (or -force-import-all) is given.

; RUN: opt -module-summary %s -o %t.main.bc
; RUN: opt -module-summary %p/Inputs/noinline.ll -o %t.inputs.noinline.bc
; RUN: llvm-lto -thinlto -o %t.summary %t.main.bc %t.inputs.noinline.bc

; RUN: opt -passes=function-import -summary-file %t.summary.thinlto.bc %t.main.bc -S 2>&1 \
; RUN:   | FileCheck -check-prefix=NOIMPORT %s
; RUN: opt -passes=function-import -import-critical-noinline -summary-file %t.summary.thinlto.bc \
; RUN:   %t.main.bc -S 2>&1 | FileCheck -check-prefix=IMPORT %s

; Without the GUID in the metadata the edge is not critical, and the option
; does not import the noinline callee.
; RUN: sed -e 's/ !prof !0//' %s | opt -module-summary -o %t.plain.bc
; RUN: llvm-lto -thinlto -o %t.plain.summary %t.plain.bc %t.inputs.noinline.bc
; RUN: opt -passes=function-import -import-critical-noinline -summary-file %t.plain.summary.thinlto.bc \
; RUN:   %t.plain.bc -S 2>&1 | FileCheck -check-prefix=NOIMPORT %s

define i32 @main() !prof !0 {
entry:
  %f = alloca i64, align 8
  call void @foo(ptr %f)
  ret i32 0
}

; NOIMPORT: declare void @foo(ptr)
; IMPORT: define available_externally void @foo
declare void @foo(ptr)

; 6699318081062747564 is the GUID of foo.
!0 = !{!"function_entry_count", i64 1, i64 6699318081062747564}
