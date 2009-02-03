; Test whether negative values > 64 bits retain their negativeness.
; RUN: llvm-as < %s | llvm-dis | grep {add i65.*, -1}

define i65 @testConsts(i65 %N) {
  %a = add i65 %N, -1
  ret i65 %a
}
