; REQUIRES: x86_64-linux
; RUN: opt -passes='print<ir2vec-vocab>' -S -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_2D_vocab.json %s 2> %t1.log 
; RUN: diff %S/Inputs/reference_default_vocab_print.txt %t1.log

; RUN: opt -passes='print<ir2vec-vocab>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_2D_vocab.json -ir2vec-opc-weight=0.5 -ir2vec-type-weight=0.5 -ir2vec-arg-weight=0.5 %s 2> %t2.log
; RUN: diff %S/Inputs/reference_wtd1_vocab_print.txt %t2.log

; RUN: opt -passes='print<ir2vec-vocab>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_2D_vocab.json -ir2vec-opc-weight=0.1 -ir2vec-type-weight=0 -ir2vec-arg-weight=0 %s 2> %t3.log
; RUN: diff %S/Inputs/reference_wtd2_vocab_print.txt %t3.log
 
define dso_local void @test() {
  entry:
    ret void
}
