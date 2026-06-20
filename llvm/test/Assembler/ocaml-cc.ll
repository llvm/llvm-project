; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; The OCaml calling conventions print as "ocamlcc" / "ocaml_cc" (cc128/cc129).

; CHECK: define ocamlcc i64 @f(
define cc128 i64 @f(i64 %x) {
  ret i64 %x
}
