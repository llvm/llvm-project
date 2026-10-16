; REQUIRES: x86_64-linux
; RUN: llc -o /dev/null -print-mir2vec-vocab -mir2vec-vocab-path=%S/Inputs/mir2vec_dummy_2D_vocab.json %s 2> %t1.log 
; RUN: diff %S/Inputs/reference_x86_vocab_print.txt %t1.log

; RUN: llc -o /dev/null -print-mir2vec-vocab -mir2vec-opc-weight=1 -mir2vec-vocab-path=%S/Inputs/mir2vec_dummy_2D_vocab.json %s 2> %t1.log 
; RUN: diff %S/Inputs/reference_x86_vocab_print.txt %t1.log

; RUN: llc -o /dev/null -print-mir2vec-vocab -mir2vec-opc-weight=0.5 -mir2vec-vocab-path=%S/Inputs/mir2vec_dummy_2D_vocab.json %s 2> %t1.log 
; RUN: diff %S/Inputs/reference_x86_vocab_wo=0.5_print.txt %t1.log

define dso_local void @test() {
  entry:
    ret void
}
