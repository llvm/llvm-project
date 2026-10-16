; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: common global [32 x b8] zeroinitializer
; CHECK: constant [1 x b8] zeroinitializer
; CHECK: constant [15 x b8] c"Hello, World!\0A\00"
; CHECK: constant [15 x b8] c"Hello, World!\0A\00"
@a = common global [32 x b8] zeroinitializer, align 1
@b = constant [1 x b8] zeroinitializer
@c = constant [15 x b8] [b8 72, b8 101, b8 108, b8 108, b8 111, b8 44, b8 32,  b8 87, b8 111, b8 114, b8 108, b8 100, b8 33,  b8 10, b8 0]
@d = constant [15 x b8] c"Hello, World!\0A\00"

define void @bytes(b1 %a, b3 %b, b5 %c, b8 %d, b16 %e, b32 %f, b64 %g, b128 %h, <8 x b5> %i, <2 x b64> %j) {
; CHECK-LABEL: define void @bytes(
; CHECK-SAME: b1 [[A:%.*]], b3 [[B:%.*]], b5 [[C:%.*]], b8 [[D:%.*]], b16 [[E:%.*]], b32 [[F:%.*]], b64 [[G:%.*]], b128 [[H:%.*]], <8 x b5> [[I:%.*]], <2 x b64> [[J:%.*]]) {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @byte_alloca() {
; CHECK-LABEL: define void @byte_alloca() {
; CHECK-NEXT:    [[B1:%.*]] = alloca b8, align 1
; CHECK-NEXT:    [[B8:%.*]] = alloca b64, align 8
; CHECK-NEXT:    [[V:%.*]] = alloca <4 x b64>, align 32
; CHECK-NEXT:    [[A:%.*]] = alloca [4 x b64], align 8
; CHECK-NEXT:    ret void
;
  %b1 = alloca b8
  %b8 = alloca b64
  %v  = alloca <4 x b64>
  %a  = alloca [4 x b64]
  ret void
}

define void @byte_load_store(ptr %ptr) {
; CHECK-LABEL: define void @byte_load_store(
; CHECK-SAME: ptr [[PTR:%.*]]) {
; CHECK-NEXT:    [[B:%.*]] = load b8, ptr [[PTR]], align 1
; CHECK-NEXT:    store b8 [[B]], ptr [[PTR]], align 1
; CHECK-NEXT:    store b8 0, ptr [[PTR]], align 1
; CHECK-NEXT:    [[V:%.*]] = load <4 x b64>, ptr [[PTR]], align 32
; CHECK-NEXT:    store <4 x b64> [[V]], ptr [[PTR]], align 32
; CHECK-NEXT:    store <4 x b64> <b64 0, b64 1, b64 2, b64 3>, ptr [[PTR]], align 32
; CHECK-NEXT:    [[A:%.*]] = load [4 x b8], ptr [[PTR]], align 1
; CHECK-NEXT:    store [4 x b8] [[A]], ptr [[PTR]], align 1
; CHECK-NEXT:    store [4 x b8] c"\00\01\02\03", ptr [[PTR]], align 1
; CHECK-NEXT:    ret void
;
  %b = load b8, ptr %ptr
  store b8 %b, ptr %ptr
  store b8 0, ptr %ptr
  %v = load <4 x b64>, ptr %ptr
  store <4 x b64> %v, ptr %ptr
  store <4 x b64> <b64 0, b64 1, b64 2, b64 3>, ptr %ptr
  %a = load [4 x b8], ptr %ptr
  store [4 x b8] %a, ptr %ptr
  store [4 x b8] [b8 0, b8 1, b8 2, b8 3], ptr %ptr
  ret void
}

define void @bitcasts(i64 %i, b64 %b, ptr %p) {
; CHECK-LABEL: define void @bitcasts(
; CHECK-SAME: i64 [[I:%.*]], b64 [[B:%.*]], ptr [[P:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast ptr [[P]] to b64
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i64 [[I]] to b64
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast b64 [[B]] to <8 x b8>
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast b64 [[B]] to i64
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast b64 [[B]] to ptr
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast <8 x b8> [[TMP3]] to <2 x b32>
; CHECK-NEXT:    [[TMP7:%.*]] = bitcast <2 x b32> [[TMP6]] to b64
; CHECK-NEXT:    [[TMP8:%.*]] = bitcast <2 x b32> splat (b32 1) to b64
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast <8 x b8> [[TMP3]] to <4 x i16>
; CHECK-NEXT:    [[TMP10:%.*]] = bitcast <2 x b32> [[TMP6]] to ptr
; CHECK-NEXT:    ret void
;
  %1 = bitcast ptr %p to b64
  %2 = bitcast i64 %i to b64
  %3 = bitcast b64 %b to <8 x b8>
  %4 = bitcast b64 %b to i64
  %5 = bitcast b64 %b to ptr
  %6 = bitcast <8 x b8> %3 to <2 x b32>
  %7 = bitcast <2 x b32> %6 to b64
  %8 = bitcast <2 x b32> <b32 1, b32 1> to b64
  %9 = bitcast <8 x b8> %3 to <4 x i16>
  %10 = bitcast <2 x b32> %6 to ptr
  ret void
}

define void @freeze(b3 %t, b64 %b, <4 x b64> %v) {
; CHECK-LABEL: define void @freeze(
; CHECK-SAME: b3 [[T:%.*]], b64 [[B:%.*]], <4 x b64> [[V:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = freeze b3 [[T]]
; CHECK-NEXT:    [[TMP2:%.*]] = freeze b64 [[B]]
; CHECK-NEXT:    [[TMP3:%.*]] = freeze <4 x b64> [[V]]
; CHECK-NEXT:    ret void
;
  %1 = freeze b3 %t
  %2 = freeze b64 %b
  %3 = freeze <4 x b64> %v
  ret void
}
