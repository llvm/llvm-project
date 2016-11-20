; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/import_opaque_type.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc

; Check that we import correctly the imported type to replace the opaque one here
; RUN: llvm-lto -thinlto-action=import %t.bc -thinlto-index=%t3.bc -o - | llvm-dis -o - | FileCheck %s


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; CHECK: %0 = type { i8 }
%0 = type opaque

%a = type { %0 * }

declare void @baz()
define void @foo(%a *) {
	call void @baz()
	ret void
}

define i32 @main() {
    call void @foo(%a *null)
	ret i32 0
}
