; RUN: llc -asm-show-inst  -march=mips64el -mcpu=mips64r6 < %s -filetype=asm -o - | FileCheck %s -check-prefix=CHK64 
; RUN: llc -asm-show-inst  -march=mipsel -mcpu=mips32r6 < %s -filetype=asm -o -| FileCheck %s -check-prefix=CHK32

define internal i32 @atomic_load_test1() #0 {
entry:

  %load_add = alloca i32*, align 8
  %.atomicdst = alloca i32, align 4
  %0 = load i32*, i32** %load_add, align 8
  %1 = load atomic i32, i32* %0 acquire, align 4
  store i32 %1, i32* %.atomicdst, align 4
  %2 = load i32, i32* %.atomicdst, align 4
  
  ret i32 %2
}

define internal i64 @atomic_load_test2() #0 {
entry:

  %load_add = alloca i64*, align 16
  %.atomicdst = alloca i64, align 8
  %0 = load i64*, i64** %load_add, align 16
  %1 = load atomic i64, i64* %0 acquire, align 8
  store i64 %1, i64* %.atomicdst, align 8
  %2 = load i64, i64* %.atomicdst, align 8
  
  ret i64 %2
}
;CHK32:  LL_R6
;CHK32:  SC_R6
;CHK64:  LLD_R6
;CHK64:  SCD_R6
