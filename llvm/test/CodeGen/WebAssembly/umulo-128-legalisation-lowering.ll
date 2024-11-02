; RUN: llc < %s -mtriple=wasm32 -wasm-keep-registers | FileCheck %s --check-prefixes=WASM32
; NOTE: did not compile on wasm64 at the time the test was created!

define { i128, i8 } @muloti_test(i128 %l, i128 %r) unnamed_addr #0 {
; WASM32-LABEL: muloti_test
; WASM32: global.get      $push18=, __stack_pointer
; WASM32: i32.const       $push19=, 48
; WASM32: i32.sub         $push40=, $pop18, $pop19
; WASM32: local.tee       $push39=, 5, $pop40
; WASM32: global.set      __stack_pointer, $pop39
; WASM32: local.get       $push41=, 5
; WASM32: i32.const       $push22=, 32
; WASM32: i32.add         $push23=, $pop41, $pop22
; WASM32: local.get       $push43=, 1
; WASM32: i64.const       $push0=, 0
; WASM32: local.get       $push42=, 3
; WASM32: i64.const       $push38=, 0
; WASM32: call            __multi3, $pop23, $pop43, $pop0, $pop42, $pop38
; WASM32: local.get       $push44=, 5
; WASM32: i32.const       $push24=, 16
; WASM32: i32.add         $push25=, $pop44, $pop24
; WASM32: local.get       $push46=, 4
; WASM32: i64.const       $push37=, 0
; WASM32: local.get       $push45=, 1
; WASM32: i64.const       $push36=, 0
; WASM32: call            __multi3, $pop25, $pop46, $pop37, $pop45, $pop36
; WASM32: local.get       $push49=, 5
; WASM32: local.get       $push48=, 2
; WASM32: i64.const       $push35=, 0
; WASM32: local.get       $push47=, 3
; WASM32: i64.const       $push34=, 0
; WASM32: call            __multi3, $pop49, $pop48, $pop35, $pop47, $pop34
; WASM32: local.get       $push51=, 0
; WASM32: local.get       $push50=, 5
; WASM32: i64.load        $push1=, 32($pop50)
; WASM32: i64.store       0($pop51), $pop1
; WASM32: local.get       $push55=, 0
; WASM32: local.get       $push52=, 5
; WASM32: i32.const       $push5=, 40
; WASM32: i32.add         $push6=, $pop52, $pop5
; WASM32: i64.load        $push33=, 0($pop6)
; WASM32: local.tee       $push32=, 3, $pop33
; WASM32: local.get       $push53=, 5
; WASM32: i64.load        $push3=, 0($pop53)
; WASM32: local.get       $push54=, 5
; WASM32: i64.load        $push2=, 16($pop54)
; WASM32: i64.add         $push4=, $pop3, $pop2
; WASM32: i64.add         $push31=, $pop32, $pop4
; WASM32: local.tee       $push30=, 1, $pop31
; WASM32: i64.store       8($pop55), $pop30
; WASM32: local.get       $push62=, 0
; WASM32: local.get       $push56=, 2
; WASM32: i64.const       $push29=, 0
; WASM32: i64.ne          $push8=, $pop56, $pop29
; WASM32: local.get       $push57=, 4
; WASM32: i64.const       $push28=, 0
; WASM32: i64.ne          $push7=, $pop57, $pop28
; WASM32: i32.and         $push9=, $pop8, $pop7
; WASM32: local.get       $push58=, 5
; WASM32: i64.load        $push10=, 8($pop58)
; WASM32: i64.const       $push27=, 0
; WASM32: i64.ne          $push11=, $pop10, $pop27
; WASM32: i32.or          $push12=, $pop9, $pop11
; WASM32: local.get       $push59=, 5
; WASM32: i64.load        $push13=, 24($pop59)
; WASM32: i64.const       $push26=, 0
; WASM32: i64.ne          $push14=, $pop13, $pop26
; WASM32: i32.or          $push15=, $pop12, $pop14
; WASM32: local.get       $push61=, 1
; WASM32: local.get       $push60=, 3
; WASM32: i64.lt_u        $push16=, $pop61, $pop60
; WASM32: i32.or          $push17=, $pop15, $pop16
; WASM32: i32.store8      16($pop62), $pop17
; WASM32: local.get       $push63=, 5
; WASM32: i32.const       $push20=, 48
; WASM32: i32.add         $push21=, $pop63, $pop20
; WASM32: global.set      __stack_pointer, $pop21

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
