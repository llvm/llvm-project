; RUN: llc < %s -mtriple=wasm32 -wasm-keep-registers | FileCheck %s --check-prefixes=WASM32
; NOTE: did not compile on wasm64 at the time the test was created!

define { i128, i8 } @muloti_test(i128 %l, i128 %r) unnamed_addr #0 {
; WASM32-LABEL: muloti_test
; WASM32: global.get      $push16=, __stack_pointer
; WASM32: i32.const       $push17=, 48
; WASM32: i32.sub         $push38=, $pop16, $pop17
; WASM32: local.tee       $push37=, 5, $pop38
; WASM32: global.set      __stack_pointer, $pop37
; WASM32: local.get       $push39=, 5
; WASM32: i32.const       $push22=, 32
; WASM32: i32.add         $push23=, $pop39, $pop22
; WASM32: local.get       $push41=, 1
; WASM32: i64.const       $push0=, 0
; WASM32: local.get       $push40=, 3
; WASM32: i64.const       $push36=, 0
; WASM32: call __multi3,  $pop23, $pop41, $pop0, $pop40, $pop36
; WASM32: local.get       $push42=, 5
; WASM32: i32.const       $push20=, 16
; WASM32: i32.add         $push21=, $pop42, $pop20
; WASM32: local.get       $push44=, 4
; WASM32: i64.const       $push35=, 0
; WASM32: local.get       $push43=, 1
; WASM32: i64.const       $push34=, 0
; WASM32: call __multi3,  $pop21, $pop44, $pop35, $pop43, $pop34
; WASM32: local.get       $push47=, 5
; WASM32: local.get       $push46=, 2
; WASM32: i64.const       $push33=, 0
; WASM32: local.get       $push45=, 3
; WASM32: i64.const       $push32=, 0
; WASM32: call __multi3,  $pop47, $pop46, $pop33, $pop45, $pop32
; WASM32: local.get       $push49=, 0
; WASM32: local.get       $push48=, 5
; WASM32: i64.load        $push1=, 32($pop48)
; WASM32: i64.store       0($pop49), $pop1
; WASM32: local.get       $push53=, 0
; WASM32: local.get       $push50=, 5
; WASM32: i64.load        $push31=, 40($pop50)
; WASM32: local.tee       $push30=, 3, $pop31
; WASM32: local.get       $push51=, 5
; WASM32: i64.load        $push3=, 0($pop51)
; WASM32: local.get       $push52=, 5
; WASM32: i64.load        $push2=, 16($pop52)
; WASM32: i64.add         $push4=, $pop3, $pop2
; WASM32: i64.add         $push29=, $pop30, $pop4
; WASM32: local.tee       $push28=, 1, $pop29
; WASM32: i64.store       8($pop53), $pop28
; WASM32: local.get       $push60=, 0
; WASM32: local.get       $push54=, 2
; WASM32: i64.const       $push27=, 0
; WASM32: i64.ne          $push6=, $pop54, $pop27
; WASM32: local.get       $push55=, 4
; WASM32: i64.const       $push26=, 0
; WASM32: i64.ne          $push5=, $pop55, $pop26
; WASM32: i32.and         $push7=, $pop6, $pop5
; WASM32: local.get       $push56=, 5
; WASM32: i64.load        $push8=, 8($pop56)
; WASM32: i64.const       $push25=, 0
; WASM32: i64.ne          $push9=, $pop8, $pop25
; WASM32: i32.or          $push10=, $pop7, $pop9
; WASM32: local.get       $push57=, 5
; WASM32: i64.load        $push11=, 24($pop57)
; WASM32: i64.const       $push24=, 0
; WASM32: i64.ne          $push12=, $pop11, $pop24
; WASM32: i32.or          $push13=, $pop10, $pop12
; WASM32: local.get       $push59=, 1
; WASM32: local.get       $push58=, 3
; WASM32: i64.lt_u        $push14=, $pop59, $pop58
; WASM32: i32.or          $push15=, $pop13, $pop14
; WASM32: i32.store8      16($pop60), $pop15
; WASM32: local.get       $push61=, 5
; WASM32: i32.const       $push18=, 48
; WASM32: i32.add         $push19=, $pop61, $pop18
; WASM32: global.set      __stack_pointer, $pop19

start:
  %0 = tail call { i128, i1 } @llvm.umul.with.overflow.i128(i128 %l, i128 %r) #2
  %1 = extractvalue { i128, i1 } %0, 0
  %2 = extractvalue { i128, i1 } %0, 1
  %3 = zext i1 %2 to i8
  %4 = insertvalue { i128, i8 } undef, i128 %1, 0
  %5 = insertvalue { i128, i8 } %4, i8 %3, 1
  ret { i128, i8 } %5
}

; Function Attrs: nounwind readnone speculatable
declare { i128, i1 } @llvm.umul.with.overflow.i128(i128, i128) #1

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }
